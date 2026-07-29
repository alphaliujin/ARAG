#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" || { echo "Failed to resolve script dir" >&2; exit 1; }
cd "$SCRIPT_DIR" || { echo "Failed to cd $SCRIPT_DIR" >&2; exit 1; }

# 一次性读取 input/output 目录,避免两个 python 子进程间出现 config 漂移。
# 旧实现: 两个独立 python -c 调用,若第二个失败 mkdir "" 会创建奇怪的目录。
read_cfg=$(python3 -c "
from x2md.config import load_config
c = load_config()
print(c.input_dir)
print(c.output_dir)
" 2>&1) || {
    echo "Error: failed to load x2md.conf" >&2
    echo "$read_cfg" >&2
    exit 1
}
input_dir=$(echo "$read_cfg" | sed -n '1p')
output_dir=$(echo "$read_cfg" | sed -n '2p')

if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
    echo "Error: x2md.conf returned empty input_dir/output_dir" >&2
    exit 1
fi

if [ ! -d "$input_dir" ]; then
    echo "Error: input directory '$input_dir' not found."
    echo "Check [x2md] input_dir in x2md.conf."
    exit 1
fi

mkdir -p "$output_dir"

echo "=== Step 1: Converting documents to Markdown ==="
echo "    Input:  $input_dir"
echo "    Output: $output_dir"
python3 -m x2md.cli

echo ""
echo "=== Step 2: Extracting image features with ViT ==="
python3 -m x2md.cli --vit-only

echo ""
echo "Done! Output files are in: $output_dir/"
