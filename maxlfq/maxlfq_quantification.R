#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

library(iq)

# 从命令行参数中获取输入文件和输出文件名
input_file <- args[1]
output_file <- args[2]
pdf_output <- args[3]
filter_pep_param <- args[4]
filter_pep_qvalue <- args[5]
filter_pro_param <- args[6]
filter_pro_qvalue <- args[7]

# 调用process_long_format函数
filter_double_less <- setNames(c(filter_pep_qvalue, filter_pro_qvalue), c(filter_pep_param, filter_pro_param))
process_long_format(input_file,
    output_filename = output_file,
    pdf_out = pdf_output,
    annotation_col = c("Protein.Names", "Genes"),
    filter_double_less = filter_double_less
)

