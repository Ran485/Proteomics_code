#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File     : utils.py
@Time     : 2023/04/21 13:26:04
@Author   : RanPeng 
@Version  : pyhton 3.8.6
@Contact  : 21112030023@m.fudan.edu.cn
@License  : (C)Copyright 2022-2023, DingLab-CHINA-SHNAGHAI
@Function : None
'''
# here put the import lib

import os
import sys
import time
import pandas as pd
import numpy as np
import config_file as cfg_file

from datetime import datetime
from functools import wraps
from rich.progress import track
from rich.console import Console
from joblib import Parallel, delayed
from rich.progress import track
from multiprocessing import cpu_count
# from plots import *


# Get the number of CPU threads
NUM_WORKERS = 10
# print(f'Calling {NUM_WORKERS} CPU threads for parallel processing.')


def change_root_dir(path=None):
    """
    change the root directory: set as the 'PhosphoPreprocessy/'
    """
    print("Current Working Directory ", os.getcwd())
    # path = '/Desktop/CCA/Metadata/phosphosite-mapping（2021-11-11）/PhosphoPreprocessy'
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))


# Create the output directory
def create_output_dir(filename=None, creat_time_subdir=False):
    """Create the output directory.

    Args:
        filename (str): A given filename.
        creat_time_subdir (bool, optional):
            creat 2021-11-12 subdirectory,defaults to True.
    Returns:
        output_dir (str): The output directory.
    """
    root_dir = "./"
    out_path = root_dir + filename
    if not os.path.isdir(out_path):  # in case root_dir doesn't exist
        os.makedirs(out_path)
        print(f"Successfully created output subdir: {out_path}")
        if creat_time_subdir:
            date_string = datetime.now().strftime("%Y_%m_%d")
            out_path = os.path.join(out_path, date_string)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
                print(f"Successfully created output subdir: {out_path}")
    else:
        print(f"The current path: {out_path} already exist")
    return out_path + "/"


# create a timer decorator
def timer(func):
    """Return the function start and finished cost time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        title_start = f" {func.__name__!r} function begins running... "
        print(f"\n{title_start.center(84, '=')}\n")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        title_end = f" {func.__name__!r} function was succesfully done in {round((end_time-start_time), 2)}s "
        print(f"\n{title_end.center(84, '=')}\n")
        return result

    return wrapper


def status_info():
    """Return a status message for processing the files."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            console = Console()
            with console.status(
                f"[bold green]Working on running {func.__name__}...",
                spinner="aesthetic",
            ) as status:
                console.log(f"[bold][blue]Processing {func.__name__} started...")
                result = func(*args, **kwargs)
                console.log(f"[bold][red]Finished!")
                return result

        return wrapper

    return decorator


def memory():
    """Check the memory usage."""
    import psutil

    mem = psutil.virtual_memory()
    zj = float(mem.total) / 1024 / 1024 / 1024
    ysy = float(mem.used) / 1024 / 1024 / 1024
    kx = float(mem.free) / 1024 / 1024 / 1024
    print("Total system memory:%d.3GB" % zj)
    print("The system has used memory:%d.3GB" % ysy)
    print("System free memory:%d.3GB" % kx)


def reduce_mem_usage(df, verbose=True):
    # List of numeric data types
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    # Starting memory usage
    start_mem = df.memory_usage().sum() / (1024**2)

    # Iterate over columns in the DataFrame
    for col in df.columns:
        # Get the dtype of the column
        col_type = df[col].dtype

        # Check if the column is numeric
        if col_type in numerics:
            # Calculate the min and max values of the column
            c_min = df[col].min()
            c_max = df[col].max()

            # Determine the optimal data type for the column
            if col_type.kind == "i":
                optimal_type = next(
                    (
                        int_type
                        for int_type in (np.int8, np.int16, np.int32, np.int64)
                        if (
                            c_min > np.iinfo(int_type).min
                            and c_max < np.iinfo(int_type).max
                        )
                    ),
                    np.int64,
                )
            else:
                optimal_type = next(
                    (
                        float_type
                        for float_type in (np.float16, np.float32, np.float64)
                        if (
                            c_min > np.finfo(float_type).min
                            and c_max < np.finfo(float_type).max
                        )
                    ),
                    np.float64,
                )
            # Convert the data type of the column to the optimal data type
            df[col] = df[col].astype(optimal_type)
    # Ending memory usage
    end_mem = df.memory_usage().sum() / (1024**2)
    # Print memory usage information
    memory_usage_info(verbose, start_mem, end_mem)
    # Return the DataFrame
    return df


def memory_usage_info(verbose, start_mem, end_mem):
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )


def index_transform(df, convert_reverse=False):
    if convert_reverse:
        df["index"] = df.index
        df = df.loc[:, ~df.columns.duplicated()]
        df[["Genes", "Stripped.Sequence", "Modified.Sequence", "Precursor.Id"]] = (
            df["index"].astype(str).str.split("_", expand=True)
        )
        df.set_index(
            ["Genes", "Stripped.Sequence", "Modified.Sequence", "Precursor.Id"],
            inplace=True,
        )
    else:
        df["index"] = (
            df["Genes"]
            + "_"
            + df["Stripped.Sequence"]
            + "_"
            + df["Modified.Sequence"]
            + "_"
            + df["Precursor.Id"]
        )
        df.set_index("index", inplace=True)
    return df


def applyParallel(dfGrouped, func):
    """Apply a function to each group in a pandas dataframe in parallel.

    Args:
        dfGrouped (DataFrameGroupBy): The pandas dataframe grouped by some column.
        func (function): The function to apply to each group.

    Returns:
        DataFrame: The result of the function applied to each group.
    """
    # print(f'Calling {NUM_WORKERS} CPU threads for parallel processing.')

    # Run the function in parallel with the specified number of workers
    res = Parallel(n_jobs=NUM_WORKERS)(
        delayed(func)(group)
        for name, group in track(dfGrouped, description="Parallel processing...")
    )
    return pd.concat(res)


def save_log(outpath='./'):
    """save log to file

    Args:
        path (str, optional): The log file outpath. Defaults to './'.
    """

    class Logger(object):
        def __init__(self, filename="Default.log", outpath="./"):
            self.terminal = sys.stdout
            self.log = open(
                os.path.join(outpath, filename),
                "a",
                encoding='utf8',
            )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.now().strftime('day_' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', outpath=outpath)

    # 这里输出之后的所有的输出的print 内容即将写入日志
    print(fileName.center(80, '='))


def data_transform(df, convert_reverse=False) -> pd.DataFrame:
        """
        Perform index transformation on the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to be transformed.
        convert_reverse : bool, optional
            Whether to perform reverse transformation, by default False.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        if convert_reverse:
            df["index"] = df.index
            df = df.loc[:, ~df.columns.duplicated()]
            df[
                [
                    "Genes",
                    "Stripped.Sequence",
                    "Modified.Sequence",
                    "Precursor.Id",
                    "file_name"
                ]
            ] = (
                df["index"].astype(str).str.split("_", expand=True)
            )
            df.set_index(
                [
                    "Genes",
                    "Stripped.Sequence",
                    "Modified.Sequence",
                    "Precursor.Id",
                    "file_name"
                ],
                inplace=True,
            )
        else:
            df["index"] = (
                df["Genes"]
                + "_"
                + df["Stripped.Sequence"]
                + "_"
                + df["Modified.Sequence"]
                + "_"
                + df["Precursor.Id"]
                + "_"
                + df["file_name"]
            )
            df.set_index("index", inplace=True)
        return df


def median_normalize(df):
    """
    Perform median normalization on a pandas DataFrame.

    :param df: pandas DataFrame to be normalized. Each column is treated as a sample, and each row is a quantity.
    :return: normalized pandas DataFrame
    """
    print("Performing median normalization ...\n")

    # Calculate the global median across all columns/samples
    global_median = df.median().median()

    # Subtract each column's median from all values in the column, and then add the global median.
    df_normalized = df.subtract(df.median()).add(global_median)

    return df_normalized


def split_n(s, delimiter, n):
    parts = s.split(delimiter)
    return parts[:n] + [delimiter.join(parts[n:])]


def preprocess(data, filter_protein_qvalue=False, outpath=None):
    if filter_protein_qvalue:
        data1 = data[data['Protein.Q.Value'] < 0.05]
    data_long = data_transform(data1)
    data_long.reset_index(inplace=True)
    data_long.rename(columns={'index':'peptide'}, inplace=True)
    # Use the pivot method to convert it to wide format
    df_wide = data_long.pivot_table(index='peptide', columns='File.Name', values='Precursor.Normalised')
    df_wide.replace(0, np.nan, inplace=True)
    df_wide = np.log2(df_wide + 1)
    df_wide_normalize = median_normalize(df_wide)
    # Melt wide format data into long format data
    df_long = df_wide_normalize.reset_index().melt(id_vars='peptide', var_name='File.Name', value_name='Precursor.Normalised')
    df_long.dropna(inplace=True)
    df_long.rename(columns={'peptide':'index'}, inplace=True)
    df_long['str'] = df_long['index'].apply(lambda x: split_n(x, '_', 4))
    df_long[["Genes", "Stripped.Sequence", "Modified.Sequence", "Precursor.Id", "file_name"]] = pd.DataFrame(df_long.str.tolist(), index= df_long.index)
    df_long.drop(columns=['str', 'index'], inplace=True)

    
    # plot the results
    print(f"Plot the density plot for raw data...\n")
    # Box and density plot before normalization
    density_plot(df_wide, title = "Density curve and Rug Plot of raw data", x_title = "LFQ Abundance [Log2]", y_title = "Probability density", outpath=outpath)
    print(f"Plot the boxplot for raw data...\n")
    boxplot(df_wide, log_transform=False, title='Box Plot of raw data', x_title='Samples', y_title='LFQ Abundance [Log2]', outpath=outpath)

    # Box and density plot after normalization
    print(f"Plot the density plot after median normalize..\n")
    density_plot(df_wide_normalize, title = "Density curve and Rug Plot after normalization", x_title = "LFQ Abundance [Log2]", y_title = "Probability density", outpath=outpath)
    print(f"Plot the density plot after median normalize...\n")
    boxplot(df_wide_normalize, log_transform=False, title='Box Plot after median normalization', x_title='Samples', y_title='LFQ Abundance [Log2]', outpath=outpath)

    return df_long
