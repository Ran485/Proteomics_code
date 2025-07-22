#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import shutil
import subprocess
import sys

def find_library(LIB_PATH):
    libs = os.listdir(LIB_PATH)
    libs = [os.path.join(LIB_PATH, lib) for lib in libs if lib.endswith('.speclib')]
    return libs

def convert_raw_files(dia_file, DIA_NN, THREADS, TMP_PATH):
    raw_dia = dia_file + '.dia'
    converted_raw_dia = os.path.join(TMP_PATH, os.path.basename(raw_dia))
    if not os.path.isfile(converted_raw_dia):
        command = f"{DIA_NN} --threads {THREADS} --convert --f {dia_file}"
        print(f"Command: {command}")
        subprocess.call(command, shell=True)
        shutil.move(raw_dia, converted_raw_dia)
        return converted_raw_dia
    elif os.path.isfile(converted_raw_dia):
        print(f"\nThe file {converted_raw_dia} already exists!")
        return converted_raw_dia
    else:
        return None

def run_diann(sample, FILE_PATH, LIB_PATH, OUT_PATH, TMP_PATH, THREADS, DIA_NN, FILE_TYPE):
    libs = find_library(LIB_PATH)
    raw_file = os.path.join(FILE_PATH, sample)
    if raw_file.endswith(FILE_TYPE):
        converted_raw_dia = convert_raw_files(raw_file, DIA_NN, THREADS, TMP_PATH)
        if converted_raw_dia:
            for lib in libs:
                tmp_dir = os.path.join(OUT_PATH, os.path.splitext(os.path.basename(raw_file))[0])
                if not os.path.isdir(tmp_dir):
                    os.mkdir(tmp_dir)
                out_file = os.path.join(tmp_dir, f"{os.path.splitext(os.path.basename(raw_file))[0]}_{os.path.splitext(os.path.basename(lib))[0]}.tsv")
                command = f"{DIA_NN} --threads {THREADS} --f {converted_raw_dia} --lib {lib} --out {out_file} --no-stats --met-excision --peak-center --int-removal 0 --qvalue 0.01 --verbose 1 --window 30 --mass-acc 20 --mass-acc-ms1 10"
                
                # 检查文件是否存在及其大小
                if os.path.isfile(out_file):
                    file_size_kb = os.path.getsize(out_file) / 1024  # 获取文件大小（KB）
                    if file_size_kb > 200:
                        print(f"File {out_file} already exists and is larger than 200KB. Skipping...")
                        continue

                print(command + '\n')
                subprocess.call(command, shell=True)

def delete_file(sample, TMP_PATH):
    dia_file = os.path.join(TMP_PATH, sample + '.dia')
    dia_quant = os.path.join(TMP_PATH, sample + '.dia.quant')
    if os.path.isfile(dia_file):
        os.remove(dia_file)
    if os.path.isfile(dia_quant):
        os.remove(dia_quant)

def validate_sample(sample, OUT_PATH):
    sample_dir = os.path.join(OUT_PATH, os.path.splitext(sample)[0])
    if os.path.exists(sample_dir):
        tsv_files = [file for file in os.listdir(sample_dir) if file.endswith('.tsv')]
        res = len(tsv_files)
        if res == 327:
            return True
    return False

if __name__ == '__main__':
    sample = sys.argv[1]
    FILE_PATH = sys.argv[2]
    LIB_PATH = sys.argv[3]
    OUT_PATH = sys.argv[4]
    TMP_PATH = sys.argv[5]
    THREADS = sys.argv[6]
    DIA_NN = sys.argv[7]
    
    print(sample)
    if not validate_sample(sample, OUT_PATH):
        run_diann(sample, FILE_PATH, LIB_PATH, OUT_PATH, TMP_PATH, THREADS, DIA_NN, ".mzML")
        delete_file(os.path.splitext(sample)[0], TMP_PATH)
    else:
        print(f"\nThe sample: {sample} has been processed!")

