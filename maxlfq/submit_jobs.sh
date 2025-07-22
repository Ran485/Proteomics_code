#!/bin/bash

# 指定目录
input_dir="/public/home/proteome/ranpeng/DataStorage/LJW_CNS/CRC_DIANN_MaxLFQ_FDR001"
output_dir="/public/home/proteome/ranpeng/DataStorage/LJW_CNS/CRC_DIANN_MaxLFQ_FDR001/LFQ_result_FDR001"
pbs_scripts_dir="/public/home/proteome/ranpeng/DataStorage/LJW_CNS/iq/pbs_scripts"

# 创建存放PBS脚本的目录，如果不存在则创建
mkdir -p "$pbs_scripts_dir"
mkdir -p "$output_dir"

# 获取指定目录下的所有 .tsv 文件
file_list=($(find "$input_dir" -maxdepth 1 -name "*.tsv"))

# 设置filter_double_less参数
filter_pep_param="Q.Value"
filter_pep_qvalue="0.01"
filter_pro_param="Protein.Q.Value"
filter_pro_qvalue="0.05"

# 遍历文件列表并生成并提交PBS脚本
for file in "${file_list[@]}"; do
  # 提取基文件名（不带路径和扩展名）
  base_name=$(basename "$file" .tsv)

  # 生成输出文件路径
  output_file="$output_dir/${base_name}_lfq_${filter_pep_param}_${filter_pep_qvalue}_${filter_pro_param}_${filter_pro_qvalue}.tsv"

  # 检查输出文件是否存在且大小大于1M
  if [ -f "$output_file" ] && [ $(stat -c%s "$output_file") -gt 1048576 ]; then
    echo "Output file $output_file already exists and is larger than 1M. Skipping job submission for $base_name."
    continue
  fi

  # 生成PBS脚本内容
  pbs_script="#!/bin/bash
#PBS -N MaxLFQ_$base_name
#PBS -l nodes=1:ppn=4
#PBS -l walltime=120:00:00
#PBS -l mem=120gb
#PBS -j oe
#PBS -o $output_dir/${base_name}_output_${filter_pep_param}_${filter_pep_qvalue}_${filter_pro_param}_${filter_pro_qvalue}.txt

# Change directory to the directory where you submitted your job
cd /public/home/proteome/ranpeng/DataStorage/LJW_CNS/iq

# Activate python environment if you have one
source activate diann

# Run your R script
Rscript maxlfq_quantification.R \"$file\" \"$output_dir/${base_name}_lfq_${filter_pep_param}_${filter_pep_qvalue}_${filter_pro_param}_${filter_pro_qvalue}.tsv\" \"$output_dir/${base_name}_lfq_${filter_pep_param}_${filter_pep_qvalue}_${filter_pro_param}_${filter_pro_qvalue}.pdf\" \"$filter_pep_param\" \"$filter_pep_qvalue\" \"$filter_pro_param\" \"$filter_pro_qvalue\""

  # 创建PBS脚本文件
  pbs_script_file="$pbs_scripts_dir/${base_name}_job.pbs"
  echo "$pbs_script" > "$pbs_script_file"

  # 提交PBS脚本
  qsub "$pbs_script_file"
done
