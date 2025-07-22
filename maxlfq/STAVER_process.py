#import pandas as pd
import os
from read_file_1 import *

# from rpy2.robjects import r
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri


# Read file and transfer to long format dataframe
def get_target_index(df, suffix):
    df["target_index"] = (
        df["Genes"]
        + "_"
        + df["Stripped.Sequence"]
        + "_"
        + df["Modified.Sequence"]
        + "_"
        + df["Precursor.Id"]
        + "_"
        + df["file_name"].str.split(suffix).str[1]
    )
    return df


def exclude_low_CI_peptide(df, low_CI_peptide_path, suffix="_F1_R1_"):
    # load original raw data
    # df = joblib_load_file(inpath=DIA_PATH)
    # add target_index
    df = get_target_index(df, suffix=suffix)
    print(f"The original data shape: {df.shape}")
    # load low CI peptides
    low_CI_peptide = pd.read_csv(low_CI_peptide_path, index_col=0)
    low_CI_peptide["taget_index"] = (
        low_CI_peptide["index"]
        + "_"
        + low_CI_peptide["file_name"].str.split(suffix).str[1]
    )
    # exclude low CI peptides
    df = df[~df["target_index"].isin(low_CI_peptide["taget_index"])]
    df.drop(columns=["target_index"], inplace=True)
    print(f"The excluded low CI peptides data shape: {df.shape}")
    return df


def main(DIA_PATH, outpath, filename):
    df = joblib_load_file(inpath=DIA_PATH)
    low_CI_peptide_path = "/public/home/proteome/ranpeng/DataStorage/LJW_CNS/iq/low_CI_peptides/low_CI_peptide_count2_cv0.2.csv"
    data = exclude_low_CI_peptide(df, low_CI_peptide_path)
    data = data[
        [
            "File.Name",
            "Genes",
            "Protein.Group",
            "Protein.Names",
            "Precursor.Id",
            "Fragment.Quant.Corrected",
            "Protein.Q.Value",
            "Q.Value",
            #"Global.Q.Value",
        ]
    ]
    data.to_csv(
        outpath + f"{filename}_merged_protein_matrix.tsv", index=False, sep="\t"
    )

    return outpath + f"{filename}_merged_protein_matrix.tsv"


def process_data(input_file, output_file, pdf_out):
    inext = importr("iq")
    rscript = f"""
    process_long_format("{input_file}",
                        output_filename = "{output_file}",
                        pdf_out = "{pdf_out}",
                        annotation_col = c("Protein.Names", "Genes"),
                        filter_double_less = c("Global.Q.Value" = "0.01", "Global.PG.Q.Value = "0.05"))
    """
    r(rscript)


if __name__ == "__main__":
    import sys

    DIA_PATH = sys.argv[1]
    outpath = sys.argv[2]
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    filename = sys.argv[3]
    outfile = main(DIA_PATH, outpath, filename)
    # MaxLFQ_outfile = outfile.replace(".tsv", "_MaxLFQ_gq0.01_pq0.05.tsv")
    # pdf_out = outfile.replace(".tsv", ".pdf")
    # process_data(outfile, MaxLFQ_outfile, pdf_out)
