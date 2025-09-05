import pandas as pd
from utils.ga import GeneticAlgorithm
from utils.data_uitl import preprocess, calcu_first_over_week



if __name__ == '__main__':
    df = pd.read_excel('附件.xlsx', sheet_name=0)
    df = preprocess(df)
    
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    print(over_week_df)