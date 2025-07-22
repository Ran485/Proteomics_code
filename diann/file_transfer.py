import os
import shutil
import pandas as pd
import argparse

def move_folders_based_on_csv(csv_file, source_dir, destination_dir, reverse=False):
    # 读取CSV文件中的实验号
    df = pd.read_csv(csv_file)
    experiment_numbers = df.iloc[:, 0].astype(str).tolist()  # 假设实验号在第一列

    # 确保目标目录存在
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if reverse:
        source_dir, destination_dir = destination_dir, source_dir

    # 迭代文件夹名称并移动匹配的文件夹
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            for exp_num in experiment_numbers:
                if exp_num in folder_name:
                    shutil.move(folder_path, os.path.join(destination_dir, folder_name))
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Move folders based on experiment numbers in a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing experiment numbers')
    parser.add_argument('source_dir', type=str, help='Path to the source directory containing folders')
    parser.add_argument('destination_dir', type=str, help='Path to the destination directory')
    parser.add_argument('--reverse', action='store_true', help='Move folders back to the original directory')

    args = parser.parse_args()
    move_folders_based_on_csv(args.csv_file, args.source_dir, args.destination_dir, args.reverse)
