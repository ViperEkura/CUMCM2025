import pandas as pd
from utils.data_uitl import preprocess, calcu_first_over_week
from utils.plot_util import plot_scatter


if __name__ == '__main__':
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    plot_scatter(over_week_df, "孕妇BMI", "Y染色体浓度")