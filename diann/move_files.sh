#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 exp_list.txt target_directory"
    exit 1
fi

# 定义实验编号列表文件和目标目录
exp_list="$1"
target_dir="$2"

# 确保目标目录存在
mkdir -p "$target_dir"

# 读取实验编号并移动相应的目录
while IFS= read -r exp_no; do
    if [ "$exp_no" != "Exp No." ]; then  # 跳过标题行
        exp_dir="${exp_no}"
        if [ -d "$exp_dir" ]; then
            mv "$exp_dir" "$target_dir"
            echo "Moved $exp_dir to $target_dir"
        else
            echo "Directory $exp_dir does not exist"
        fi
    fi
done < "$exp_list"
