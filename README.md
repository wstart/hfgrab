# hfgrab

从 HuggingFace 镜像下载模型，**下载卡住会自动发现并续传**。

```bash
hfgrab mlx-community/Qwen3.6-35B-A3B-8bit
hfgrab https://hf-mirror.com/Qwen/Qwen3.6-35B-A3B -o ~/models
```

## 为什么写这个

下大模型时反复遇到同一种故障：进程还活着、CPU 也有占用，但**连接已经归零、
一个字节都不再落盘**。官方 `hf` CLI 和 `hfd.sh` 都只看进程存活，于是卡死
被当成正常，眼睁睁等几个小时才发现进度条一直停在 22%。

hfgrab 以「目录体积是否增长」判断存活，停滞超过阈值就杀掉重启续传：

```
   22.4% 8.4GB/37.7GB  0.0B/s
  ! 120 秒无进展，重启续传（第 1/10 次）
   22.6% 8.5GB/37.7GB  47.2MB/s
```

## 特点

- **卡死自愈** —— 按实际落盘量判断，不被「进程还在」骗过
- **走镜像** —— 默认 hf-mirror.com，国内直连常被限速到卡死
- **粘贴即用** —— 从浏览器地址栏复制的链接直接能用，各种形态都认
- **断点续传** —— 中断后重跑自动跳过已完整的文件
- **多线程** —— aria2c 驱动，实测比单连接快一个数量级

## 安装

```bash
brew install aria2                    # 唯一依赖
curl -O https://raw.githubusercontent.com/wstart/hfgrab/main/hfgrab.py
chmod +x hfgrab.py && sudo mv hfgrab.py /usr/local/bin/hfgrab
```

或者直接跑，不装：

```bash
python3 hfgrab.py <repo>
```

只需 Python 3.9+ 与 aria2c，无 pip 依赖。

## 用法

```
hfgrab <repo> [选项]

  -o, --output DIR      下载目录（默认当前目录）
  -x, --threads N       每服务器连接数（默认 8）
  -j, --jobs N          并发文件数（默认 5）
      --dataset         下载数据集而非模型
      --token TOKEN     HF token，也可用 $HF_TOKEN
      --stall SECONDS   判定卡死的无进展秒数（默认 120）
      --max-restarts N  卡死后最多重启次数（默认 10）
```

仓库参数支持这些写法，都指向同一个仓库：

```
mlx-community/Qwen3.6-35B-A3B-8bit
https://hf-mirror.com/mlx-community/Qwen3.6-35B-A3B-8bit
https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit/tree/main
hf-mirror.com/models/mlx-community/Qwen3.6-35B-A3B-8bit
带 ?library=mlx 之类查询串的链接
```

换镜像源：

```bash
HF_ENDPOINT=https://huggingface.co hfgrab <repo>
```

## 几个说明

**镜像只承载文件下载。** 元数据查询（文件列表、体积）走 `/api/*`，
hf-mirror 对这类请求只做 308 重定向回官方，所以查询实际仍是官方响应；
镜像不可达时自动回退直连。真正吃带宽、真正容易卡死的是文件下载，
那部分确实走镜像。

**卡死阈值怎么定。** 默认 120 秒。调太小会在正常的慢速期误杀重启，
反而更慢；调太大则白等。网络不稳时可以 `--stall 60`。

**已完整的文件按体积跳过**，不做校验和比对——HF 的 API 给的是文件大小，
逐个算 SHA 对几十 GB 的模型代价太高。如果怀疑文件损坏，删掉重下。

## 许可

MIT
