#!/bin/bash
# use conda
# file: DIA_NN.pbs
#PBS -N DIA-NN
#PBS -l nodes=1:ppn=1
##PBS -l mem=10gb
#PBS -l walltime=72:00:00
#PBS -q default
#PBS -j oe
# import all environments virables
#PBS -V

software="/public/home/proteome/ranpeng/software"
mono_sif="/public/home/proteome/ranpeng/software/mono.sif"
ThermoRawFileParser="/public/home/proteome/ranpeng/software/ThermoRawFileParser1/ThermoRawFileParser.exe"

input_dir="/public/home/proteome/ranpeng/DataStorage/LJW_CNS/AIS"
outout_dir="/public/home/proteome/ranpeng/DataStorage/LJW_CNS/AIS/test"

singularity exec --bind /public/group_share_data/proteome/ranpeng/:/public/group_share_data/proteome/ranpeng/ $mono_sif mono $ThermoRawFileParser -i $input_dir/$sample -m 1

