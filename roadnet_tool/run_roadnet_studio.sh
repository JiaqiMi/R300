#!/bin/bash
# ============================================================
# RoadNet Studio 启动脚本
# 用法：双击运行，或在终端中 ./run_roadnet_studio.sh
# ============================================================

# 自动进入项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活 conda 环境（根据实际 conda 路径调整）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
fi

# 激活项目环境
conda activate samroad310 2>/dev/null || {
    echo "[WARNING] 无法激活 conda 环境 samroad310，尝试使用系统 Python..."
}

# 运行 GUI
echo "========================================"
echo "  RoadNet Studio - 无人车路网生成与编辑系统"
echo "========================================"
echo ""

python main_gui.py "$@"

echo ""
echo "----------------------------------------"
read -p "程序已退出，按 Enter 关闭窗口..."
