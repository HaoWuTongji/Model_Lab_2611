#!/bin/bash
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python main.py
elif command -v python3 >/dev/null 2>&1; then
  python3 main.py
else
  echo "[ERROR] 未找到 python3，请先安装 Python 3。"
  read -n 1 -s -r -p "按任意键关闭..."
fi
