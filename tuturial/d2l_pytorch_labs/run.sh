#!/usr/bin/env bash
# ============================================================
#  d2l 课程一键训练脚本（amax / 7x V100S）
#  用法：./run.sh image resnet18
#        ./run.sh text bilstm
#        ./run.sh vae
#        ./run.sh gan
#        ./run.sh list          # 看所有可用任务名
# ============================================================
set -euo pipefail

# ---------- 固定配置 ----------
PROJECT_DIR="/data1/yxy/projects/d2l_pytorch_labs"
RUNS_ROOT="/data1/yxy/runs"
LOGS_ROOT="/data1/yxy/runs/logs"
CONDA_ENV="/data1/yxy/env/d2l"
WORKERS=8

usage() {
    cat <<'EOF'
用法: ./run.sh <任务> <模型> [额外参数...]

任务:
  image     图像分类 (CIFAR-10)
              模型: smallcnn | alexnet | vgg11 | resnet18
  text      文本分类 (IMDB，需先 python prepare_data.py --dataset imdb)
              模型: bilstm | textcnn
  vae       变分自编码器 (CIFAR-10)
  gan       生成对抗网络 (CIFAR-10)
  list      列出所有可用任务
  env       打印当前环境与 GPU 状态

示例:
  ./run.sh image resnet18
  ./run.sh image vgg11 --epochs 30 --batch-size 64
  ./run.sh text bilstm --epochs 8
  ./run.sh vae --latent-dim 32
  ./run.sh gan --epochs 100

说明:
  - 自动挑选"显存占用最低"的 GPU
  - 训练在后台运行，SSH 断开不影响
  - 日志: /data1/yxy/runs/logs/<名字>.log
  - 查看: ./run.sh logs <名字>   (或 tail -f 日志路径)
  - 停止: ./run.sh stop <PID或名字>
EOF
}

# ---------- 工具函数 ----------
die() { echo "错误: $*" >&2; exit 1; }

activate_env() {
    # 不走 conda activate，直接用环境里的解释器，避免 shell 初始化依赖
    [[ -x "$CONDA_ENV/bin/python" ]] || die "找不到 Python 环境: $CONDA_ENV"
    export PATH="$CONDA_ENV/bin:$PATH"
}

pick_gpu() {
    # 选显存占用最低的卡
    local idx
    idx=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
    [[ -n "$idx" ]] || die "无法读取 GPU 状态"
    echo "$idx"
}

# ---------- 主逻辑 ----------
[[ $# -ge 1 ]] || { usage; exit 1; }
TASK="$1"; shift

case "$TASK" in
    -h|--help|help) usage; exit 0 ;;
    list)
        echo "image : smallcnn | alexnet | vgg11 | resnet18"
        echo "text  : bilstm | textcnn"
        echo "vae   : (无模型名)"
        echo "gan   : (无模型名)"
        exit 0 ;;
    env)
        activate_env
        echo "Python: $(which python)"
        python -c "import torch;print('torch:',torch.__version__,'| CUDA:',torch.cuda.is_available(),'| GPUs:',torch.cuda.device_count())"
        echo "--- GPU 占用 ---"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
        exit 0 ;;
    logs)
        [[ $# -ge 1 ]] || die "用法: ./run.sh logs <名字>"
        LOG="$LOGS_ROOT/$1.log"
        [[ -f "$LOG" ]] || die "日志不存在: $LOG"
        tail -f "$LOG" ;;
    stop)
        [[ $# -ge 1 ]] || die "用法: ./run.sh stop <PID>"
        if [[ "$1" =~ ^[0-9]+$ ]]; then
            kill "$1" && echo "已发送终止信号: PID $1"
        else
            pkill -f "$1" && echo "已终止匹配 '$1' 的进程"
        fi
        exit 0 ;;
esac

cd "$PROJECT_DIR"
activate_env
mkdir -p "$RUNS_ROOT" "$LOGS_ROOT"

# GPU：允许用环境变量 GPU=2 覆盖自动选择
GPU="${GPU:-$(pick_gpu)}"
echo "使用 GPU: cuda:$GPU"

case "$TASK" in
    image)
        MODEL="${1:-resnet18}"; shift || true
        NAME="image_${MODEL}_$(date +%m%d_%H%M)"
        CMD=(python -u train_image.py --model "$MODEL" --epochs 20 --batch-size 128
             --workers "$WORKERS" --device "cuda:$GPU" --run-dir "$RUNS_ROOT/$NAME")
        ;;
    text)
        MODEL="${1:-bilstm}"; shift || true
        NAME="text_${MODEL}_$(date +%m%d_%H%M)"
        CMD=(python -u train_text.py --model "$MODEL" --epochs 8
             --workers "$WORKERS" --device "cuda:$GPU" --run-dir "$RUNS_ROOT/$NAME")
        ;;
    vae)
        NAME="vae_$(date +%m%d_%H%M)"
        CMD=(python -u train_vae.py --epochs 30 --batch-size 128
             --workers "$WORKERS" --device "cuda:$GPU" --run-dir "$RUNS_ROOT/$NAME")
        ;;
    gan)
        NAME="gan_$(date +%m%d_%H%M)"
        CMD=(python -u train_gan.py --epochs 50 --batch-size 128
             --workers "$WORKERS" --device "cuda:$GPU" --run-dir "$RUNS_ROOT/$NAME")
        ;;
    *)
        usage; die "未知任务: $TASK" ;;
esac

# 追加用户额外参数（会覆盖同名默认值）
if [[ $# -gt 0 ]]; then
    CMD+=("$@")
fi

LOG="$LOGS_ROOT/$NAME.log"
echo "运行目录: $RUNS_ROOT/$NAME"
echo "日志文件: $LOG"
echo "命令: ${CMD[*]}"
echo "----------------------------------------"

nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
echo "已在后台启动，PID: $PID"
echo "查看进度: ./run.sh logs $NAME"
echo "或:       tail -f $LOG"
echo "停止:     ./run.sh stop $PID"