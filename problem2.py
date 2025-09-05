import pandas as pd
from utils.data_uitl import preprocess, calcu_first_over_week


if __name__ == '__main__':
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    over_week = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week.to_excel("over_week_t004.xlsx")