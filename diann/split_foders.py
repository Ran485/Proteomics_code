import os
import shutil
import argparse

def move_folders(input_dir, output_dir, new_folder_prefix, num_new_folders):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all folders in the input directory
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    # Sort the folders
    folders.sort()

    # Calculate the number of folders per new directory
    folders_per_new_dir = len(folders) // num_new_folders
    extra_folders = len(folders) % num_new_folders

    current_index = 0

    for i in range(num_new_folders):
        new_folder_name = f"{new_folder_prefix}_{i+1}"
        new_folder_path = os.path.join(output_dir, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Calculate the number of folders to move to this new directory
        num_folders_to_move = folders_per_new_dir + (1 if i < extra_folders else 0)

        for _ in range(num_folders_to_move):
            if current_index < len(folders):
                folder_to_move = folders[current_index]
                src_path = os.path.join(input_dir, folder_to_move)
                dest_path = os.path.join(new_folder_path, folder_to_move)

                # Check to avoid moving directory into itself
                if os.path.commonpath([src_path]) != os.path.commonpath([new_folder_path]):
                    shutil.move(src_path, dest_path)
                current_index += 1

def revert_folders(output_dir, new_folder_prefix, num_new_folders, original_dir):
    for i in range(num_new_folders):
        new_folder_name = f"{new_folder_prefix}_{i+1}"
        new_folder_path = os.path.join(output_dir, new_folder_name)

        if os.path.exists(new_folder_path):
            for folder in os.listdir(new_folder_path):
                src_path = os.path.join(new_folder_path, folder)
                dest_path = os.path.join(original_dir, folder)
                shutil.move(src_path, dest_path)
            # Remove the empty new folder directory
            os.rmdir(new_folder_path)

def main():
    parser = argparse.ArgumentParser(description="Move folders to new directories or revert them back.")
    parser.add_argument("input_dir", type=str, help="Input directory containing folders to move.")
    parser.add_argument("output_dir", type=str, help="Output directory where new folders will be created.")
    parser.add_argument("new_folder_prefix", type=str, help="Prefix for new folders.")
    parser.add_argument("num_new_folders", type=int, help="Number of new folders to create.")
    parser.add_argument("--revert", action="store_true", help="Revert folders back to the original directory.")
    parser.add_argument("--original_dir", type=str, help="Original directory to move folders back to when reverting.")

    args = parser.parse_args()

    if args.revert:
        if not args.original_dir:
            print("Error: You must specify the original directory with --original_dir when using --revert.")
        else:
            revert_folders(args.output_dir, args.new_folder_prefix, args.num_new_folders, args.original_dir)
    else:
        move_folders(args.input_dir, args.output_dir, args.new_folder_prefix, args.num_new_folders)

if __name__ == "__main__":
    main()
