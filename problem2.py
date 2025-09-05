import numpy as np
import pandas as pd
from utils.ga import GeneticAlgorithm
from utils.data_uitl import preprocess, calcu_first_over_week

def get_params(df: pd.DataFrame):
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    y, t, bmi = over_week_df["Y染色体浓度"].values, over_week_df["检测孕周"].values, over_week_df["孕妇BMI"].values
    N = len(df)
    
    return N, y, t, bmi

def calcu_N(ind: np.ndarray, bmi: np.ndarray) -> np.ndarray:
    """
    input: shape ind(n + 1), bmi(N), y(N)
    output: shape Ni(n)
    """
    n = len(ind) - 1
    
    Ni = np.zeros(n)
    for i in range(n):
        start_bmi = ind[i]
        end_bmi = ind[i + 1]
        mask = np.where((bmi >= start_bmi) & (bmi < end_bmi), 1, 0)
        Ni[i] = np.sum(mask)
    
    return Ni

def calcu_P(ind: np.ndarray, bmi: np.ndarray, y) -> float:
    """
    input shape: ind(n + 1), bmi(N)
    output: P (float)
    """
    n = len(ind) - 1
    N = len(bmi)
    P = np.zeros(n)
    
    for i in range(n):
        start_bmi = ind[i]
        end_bmi = ind[i + 1]
        index_mask = np.where((bmi >= start_bmi) & (bmi < end_bmi))
        P[i] = np.sum() = np.sum(y[index_mask] > 0.04)
    
    P = P / N
    
    return P

def init_func(bmi:np.ndarray, n_feature: int):
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    population = np.random.uniform(bmi_min, bmi_max, (n_feature,))
    population = np.sort(population, axis=-1)
    
    return population

def valid_func(ind: np.ndarray, bmi: np.ndarray, y: np.ndarray, n_feature: int):
    pass


if __name__ == '__main__':
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    N, y, t, bmi = get_params(df)
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    
    model = GeneticAlgorithm()
