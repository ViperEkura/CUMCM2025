import pandas as pd
from utils.data_uitl import preprocess
from utils.plot_util import plot_distribution


if __name__ == "__main__":
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    plot_distribution(df, '孕妇BMI')