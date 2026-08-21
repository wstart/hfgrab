#!/usr/bin/env python3
"""hfgrab —— 从 HuggingFace 镜像下载模型，带卡死检测与断点续传。

写这个是因为官方 hf CLI 和 hfd.sh 都有同一个盲点：下载卡住时进程还活着、
CPU 也有占用，但连接已经归零、一个字节都不再落盘。只看进程存活会把卡死
当成正常，眼睁睁等几小时。

这里以「目录体积是否增长」判断存活，停滞超过阈值就自动重启续传。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

__version__ = "1.0.0"

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OFFICIAL = "https://huggingface.co"

C = {"r": "\033[0;31m", "g": "\033[0;32m", "y": "\033[1;33m",
     "b": "\033[0;34m", "d": "\033[2m", "B": "\033[1m", "_": "\033[0m"}
if not sys.stdout.isatty():
    C = dict.fromkeys(C, "")


def say(msg: str) -> None:  print(f"{C['B']}▸ {msg}{C['_']}")
def ok(msg: str) -> None:   print(f"  {C['g']}✓{C['_']} {msg}")
def warn(msg: str) -> None: print(f"  {C['y']}!{C['_']} {msg}")
def dim(msg: str) -> None:  print(f"  {C['d']}{msg}{C['_']}")
def die(msg: str) -> None:  print(f"  {C['r']}✗ {msg}{C['_']}"); sys.exit(1)


# ---------------- 仓库标识 ----------------

def parse_repo_id(text: str) -> str:
    """从粘贴的内容里提取 owner/name。

    支持从镜像站或官网复制的各种形态：完整 URL、带 /tree/main 的页面地址、
    带查询串的链接，以及裸的 owner/name。
    """
    t = (text or "").strip().strip('"\'')
    if not t:
        return ""
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^(hf-mirror\.com|huggingface\.co|hf\.co)/", "", t)
    t = re.split(r"[?#]", t)[0]
    t = re.sub(r"^(models|datasets)/", "", t)
    t = re.sub(r"/(tree|blob|resolve|commits?|discussions|settings)(/.*)?$", "", t)
    parts = [p for p in t.strip("/").split("/") if p]
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else (parts[0] if parts else "")


def api_get(path: str, timeout: int = 25) -> Any:
    """依次尝试镜像与官方。镜像对 /api/* 多半只做 308 重定向，
    urlopen 会自动跟随；镜像不可达时回退官方。"""
    last: Exception | None = None
    for host in (MIRROR, OFFICIAL):
        try:
            req = urllib.request.Request(
                host + path, headers={"User-Agent": f"hfgrab/{__version__}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last = e
    raise last or RuntimeError("无法访问 HuggingFace API")


def repo_info(repo: str, is_dataset: bool = False) -> dict[str, Any]:
    kind = "datasets" if is_dataset else "models"
    d = api_get(f"/api/{kind}/{urllib.parse.quote(repo)}?blobs=true")
    files = [
        {"path": s["rfilename"], "size": s.get("size") or 0}
        for s in d.get("siblings", [])
        if not s["rfilename"].endswith("/")
    ]
    return {
        "id": repo,
        "files": files,
        "total": sum(f["size"] for f in files),
        "downloads": d.get("downloads", 0),
        "sha": d.get("sha", "main"),
    }


# ---------------- 下载 ----------------

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def build_aria_input(repo: str, files: list[dict], dest: Path,
                     revision: str, is_dataset: bool, token: str | None) -> Path:
    """生成 aria2c 输入清单：每个文件一行 URL + 存放位置。"""
    kind = "datasets/" if is_dataset else ""
    lines: list[str] = []
    for f in files:
        rel = f["path"]
        out = dest / rel
        if out.exists() and f["size"] and out.stat().st_size == f["size"]:
            continue                                   # 已完整，跳过
        url = f"{MIRROR}/{kind}{repo}/resolve/{revision}/{urllib.parse.quote(rel)}"
        lines.append(url)
        lines.append(f"  dir={out.parent}")
        lines.append(f"  out={out.name}")
        if token:
            lines.append(f"  header=Authorization: Bearer {token}")
    listing = dest / ".hfgrab.aria2.txt"
    listing.write_text("\n".join(lines) + "\n")
    return listing


def run_aria(listing: Path, dest: Path, threads: int, jobs: int) -> subprocess.Popen:
    cmd = [
        "aria2c", "--input-file", str(listing),
        f"--max-connection-per-server={threads}",
        f"--max-concurrent-downloads={jobs}",
        "--split=8", "--min-split-size=20M",
        "--continue=true", "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--console-log-level=warn", "--summary-interval=0",
        # 连接层面的自愈；卡死检测在外层按体积增长兜底
        "--max-tries=5", "--retry-wait=3",
        "--connect-timeout=20", "--timeout=60",
        "--lowest-speed-limit=1K",
    ]
    log = open(dest / ".hfgrab.log", "ab")
    return subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL)


def download(repo: str, dest_root: Path, *, threads: int, jobs: int,
             is_dataset: bool, token: str | None, stall_limit: int,
             max_restarts: int) -> int:
    say(f"查询 {repo}")
    try:
        info = repo_info(repo, is_dataset)
    except Exception as e:  # noqa: BLE001
        die(f"查询失败：{e}")
        return 1

    if not info["files"]:
        die("仓库里没有文件")
    ok(f"{len(info['files'])} 个文件，共 {human(info['total'])}"
       + (f"，下载量 {info['downloads']:,}" if info["downloads"] else ""))

    dest = dest_root / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    free = shutil.disk_usage(dest).free
    if free < info["total"] * 1.05:
        warn(f"磁盘可用 {human(free)}，可能不足（需 {human(info['total'])}）")

    say(f"下载到 {dest}")
    dim(f"镜像 {MIRROR} · aria2c {threads} 线程 × {jobs} 并发")

    restarts = 0
    start = time.time()
    last_print = [0.0]
    last_size = dir_size(dest)
    if last_size:
        dim(f"发现已有 {human(last_size)}，断点续传")

    while True:
        listing = build_aria_input(repo, info["files"], dest,
                                   info["sha"], is_dataset, token)
        if not listing.read_text().strip():
            break                                       # 全部已完整
        proc = run_aria(listing, dest, threads, jobs)

        stalled_since = time.time()
        while proc.poll() is None:
            time.sleep(5)
            size = dir_size(dest)
            now = time.time()
            if size > last_size:
                delta = size - last_size
                last_size = size
                stalled_since = now
                pct = size / info["total"] * 100 if info["total"] else 0
                inst = delta / 5.0          # 本轮 5 秒的瞬时速度，比平均值更能反映当下
                line = (f"  {C['b']}{pct:5.1f}%{C['_']} "
                        f"{human(size)}/{human(info['total'])}  {human(inst)}/s")
                if sys.stdout.isatty():
                    sys.stdout.write("\r" + line + "      ")
                    sys.stdout.flush()
                elif now - last_print[0] > 30:   # 非终端时每 30 秒打一行，不刷屏
                    print(line, flush=True)
                    last_print[0] = now
            elif now - stalled_since > stall_limit:
                # 这是关键：进程还活着不代表在下载
                print()
                warn(f"{stall_limit} 秒无进展，重启续传"
                     f"（第 {restarts + 1}/{max_restarts} 次）")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                restarts += 1
                if restarts > max_restarts:
                    print()
                    die(f"重启 {max_restarts} 次仍无进展，请检查网络")
                break

        if proc.poll() is not None:
            code = proc.returncode
            size = dir_size(dest)
            if info["total"] and size >= info["total"] * 0.995:
                break
            if code == 0:
                break
            print()
            restarts += 1
            if restarts > max_restarts:
                die(f"aria2c 退出码 {code}，重试次数已用尽")
            warn(f"aria2c 退出码 {code}，续传重试（{restarts}/{max_restarts}）")
            time.sleep(3)

    print()
    listing.unlink(missing_ok=True)
    final = dir_size(dest)
    elapsed = time.time() - start
    ok(f"完成：{human(final)}，耗时 {elapsed / 60:.1f} 分钟"
       f"，均速 {human(final / max(1, elapsed))}/s")
    dim(str(dest))
    return 0


# ---------------- CLI ----------------

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="hfgrab",
        description="从 HuggingFace 镜像下载模型/数据集，带卡死检测与断点续传",
        epilog="示例：\n"
               "  hfgrab mlx-community/Qwen3.6-35B-A3B-8bit\n"
               "  hfgrab https://hf-mirror.com/Qwen/Qwen3.6-35B-A3B -o ~/models\n"
               "  hfgrab openai/gsm8k --dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="仓库 ID，或直接粘贴模型页链接")
    ap.add_argument("-o", "--output", default=".", help="下载目录（默认当前目录）")
    ap.add_argument("-x", "--threads", type=int, default=8, help="每服务器连接数（默认 8）")
    ap.add_argument("-j", "--jobs", type=int, default=5, help="并发文件数（默认 5）")
    ap.add_argument("--dataset", action="store_true", help="下载数据集而非模型")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token（私有仓库或提速，也可用 $HF_TOKEN）")
    ap.add_argument("--stall", type=int, default=120,
                    help="判定卡死的无进展秒数（默认 120）")
    ap.add_argument("--max-restarts", type=int, default=10,
                    help="卡死后最多重启次数（默认 10）")
    ap.add_argument("-V", "--version", action="version", version=f"hfgrab {__version__}")
    a = ap.parse_args()

    if not shutil.which("aria2c"):
        die("需要 aria2c：brew install aria2（或 apt install aria2）")

    repo = parse_repo_id(a.repo)
    if "/" not in repo:
        die(f"无法识别仓库：{a.repo}\n     需要 owner/name 形式，或粘贴模型页链接")

    try:
        return download(repo, Path(a.output).expanduser().resolve(),
                        threads=a.threads, jobs=a.jobs, is_dataset=a.dataset,
                        token=a.token, stall_limit=a.stall,
                        max_restarts=a.max_restarts)
    except KeyboardInterrupt:
        print()
        warn("已中断，断点保留，重新运行即可续传")
        return 130


if __name__ == "__main__":
    sys.exit(main())
