#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

library(iq)
library(impute)

# 从命令行参数中获取输入文件和输出文件名
input_file <- args[1]
output_file <- args[2]
pdf_output <- args[3]
filter_pep_param <- args[4]
filter_pep_qvalue <- args[5]
filter_pro_param <- args[6]
filter_pro_qvalue <- args[7]

# 读取数据
data <- read.table(input_file, header = TRUE, sep = "\t", stringsAsFactors = FALSE)

# 查找并打印NA值的位置
na_positions <- which(is.na(data), arr.ind = TRUE)
cat("NA positions:\n")
print(na_positions)

# 检查缺失值
missing_values_count <- sum(is.na(data))
cat("Missing values count:", missing_values_count, "\n")

# 删除包含NA值的行
data_cleaned <- data[complete.cases(data), ]

# 保存清理后的数据到临时文件
temp_input_file <- tempfile(fileext = ".txt")
write.table(data_cleaned, temp_input_file, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)

# 调用process_long_format函数
filter_double_less <- setNames(c(filter_pep_qvalue, filter_pro_qvalue), c(filter_pep_param, filter_pro_param))
process_long_format(temp_input_file,
    output_filename = output_file,
    pdf_out = pdf_output,
    annotation_col = c("Protein.Names", "Genes"),
    filter_double_less = filter_double_less
)

