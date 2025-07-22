#!/bin/bash

# 定义文件路径
sample_file="CRC_nc_sample.csv"
result_file="CRC-NC-res.csv"

# 去掉LJW_sample.csv中每一行末尾的".raw"，并存储到一个临时文件
sed 's/\.raw$//' "$sample_file" > sample_no_extension.csv

# 使用grep找到在LJW_sample.csv但不在LJW_sample_result.csv中的行
grep -F -v -f "$result_file" sample_no_extension.csv > diff_output.txt

# 打印结果
echo "差集如下："
cat diff_output.txt

# 清理临时文件
rm sample_no_extension.csv

