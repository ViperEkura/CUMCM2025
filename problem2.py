import numpy as np
import pandas as pd
from utils.ga import GeneticAlgorithm
from utils.data_uitl import preprocess, calcu_first_over_week

def get_params(df: pd.DataFrame):
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    y, t, bmi = over_week_df["Y染色体浓度"].values, over_week_df["检测孕周"].values, over_week_df["孕妇BMI"].values
    return y, t, bmi

def calcu_Ni(ind: np.ndarray, bmi: np.ndarray) -> np.ndarray:
    """
    input: shape ind(2n + 1), bmi(N), y(N)
    output: shape Ni(n)
    """
    n_seg = (np.size(ind) - 1) // 2
    bmi_div = ind[:n_seg + 1]
    Ni = np.zeros(n_seg)
    
    for i in range(n_seg):
        start_bmi = bmi_div[i]
        end_bmi = bmi_div[i + 1]
        mask = np.where((bmi >= start_bmi) & (bmi < end_bmi), 1, 0)
        Ni[i] = np.sum(mask)
    
    return Ni

def calcu_Pi(ind: np.ndarray, bmi: np.ndarray, y) -> np.ndarray:
    """
    input shape: ind(2n + 1), bmi(N)
    output: P (float)
    """
    n_seg = (np.size(ind) - 1) // 2
    Pi = np.zeros(n_seg)
    bmi_div = ind[:n_seg + 1]
    
    for i in range(n_seg):
        start_bmi = bmi_div[i]
        end_bmi = bmi_div[i + 1]
        index_mask = np.where((bmi >= start_bmi) & (bmi < end_bmi))
        Pi[i] = np.sum(y[index_mask] > 0.04)
    
    Pi = Pi / np.size(y)
    
    return Pi

def valid_func(ind: np.ndarray, bmi: np.ndarray, y: np.ndarray):
    Pi = calcu_Pi(ind, bmi, y)
    Ni = calcu_Ni(ind, bmi)
    n_seg = (np.size(ind) - 1) // 2
    t = ind[n_seg + 1:]

    if np.any(Pi < 0.95):
        return False
    
    if np.any(Ni <= 25):
        return False
    
    if np.any(t < 10) or np.any(t > 25):
        return False
    
    return True

def init_func(bmi:np.ndarray, y: np.ndarray, n_seg: int):
    """
    ind[0: n_seg + 1] -> bmi_div
    ind[n_seg + 1: 2n_seg + 1] -> t
    """
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    bmi_div = np.random.uniform(bmi_min, bmi_max, (n_seg + 1))
    bmi_div = np.sort(bmi_div, axis=-1)
    bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)

    t = np.random.uniform(10, 25, (n_seg))
    ind = np.concatenate([bmi_div, t])
    
    while not valid_func(ind, bmi, y):
        bmi_div = np.random.uniform(bmi_min, bmi_max, (n_seg + 1))
        bmi_div = np.sort(bmi_div, axis=-1)
        bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)

        t = np.random.uniform(10, 25, (n_seg))
        ind = np.concatenate([bmi_div, t])
    
    return ind




if __name__ == '__main__':
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    y, t, bmi = get_params(df)
    init_sol = init_func(bmi, y, n_seg=10)

    
