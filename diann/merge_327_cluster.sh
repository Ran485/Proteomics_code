#!/bin/bash
#PBS -N MaxLFQ
#PBS -l nodes=1:ppn=4
#PBS -l walltime=120:00:00
#PBS -l mem=74gb
#PBS -j oe
##PBS -o /public/home/proteome/tangss/tiantan/qc_exp_maxlfq.txt

# Change directory to the directory where you submitted your job
cd /public/home/proteome/ranpeng/DataStorage/LJW_CNS/iq/

# Activate python environment if you have one
source activate diann

# Run your python script
python ./STAVER_process.py ../CRC_Results_327/ ../CRC_DIANN_MaxLFQ_FDR001/ CRC_updated

conda deactivate
