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

def valid_func(ind: np.ndarray, bmi: np.ndarray):
    Ni = calcu_Ni(ind, bmi)
    n_seg = (np.size(ind) - 1) // 2
    t = ind[n_seg + 1:]

    if np.any(Ni <= 25):
        return False
    
    if np.any(t < 10) or np.any(t > 25):
        return False
    
    return True

def init_sol_func(bmi:np.ndarray, n_seg: int):
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
    
    while not valid_func(ind, bmi):
        bmi_div = np.random.uniform(bmi_min, bmi_max, (n_seg + 1))
        bmi_div = np.sort(bmi_div, axis=-1)
        bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)

        t = np.random.uniform(10, 25, (n_seg))
        ind = np.concatenate([bmi_div, t])
    
    return ind

def crossover_func(parent1: np.ndarray, parent2: np.ndarray, bmi: np.ndarray, max_attempts: int = 20) -> np.ndarray:
    n_seg = (len(parent1) - 1) // 2
    available_bmi_p = n_seg - 1                         # 可用的BMI交叉点（排除固定的首尾）
    available_t_p = n_seg                               # 可用的t交叉点
    n_points = min(3, available_bmi_p, available_t_p)   # 选择较少的交叉点数量，确保不超过可用位置
    
    for _ in range(max_attempts):
        child = np.copy(parent1)
        
        # 交叉BMI分界点部分（排除固定的首尾）
        if available_bmi_p > 0:
            bmi_crossover_points = np.random.choice(range(1, n_seg), size=n_points, replace=True)
            for point in bmi_crossover_points:
                child[point] = parent2[point]
        
        # 交叉t部分
        t_crossover_points = np.random.choice(range(n_seg), size=n_points, replace=True)
        for point in t_crossover_points:
            child[n_seg + 1 + point] = parent2[n_seg + 1 + point]
        
        # 确保BMI分界点有序
        child[:n_seg+1] = np.sort(child[:n_seg+1])
        # 确保边界条件
        child[0] = np.min(bmi)
        child[n_seg] = np.max(bmi)
        
        # 确保t值在有效范围内
        t_values = child[n_seg + 1:]
        t_values = np.clip(t_values, 10, 25)
        child[n_seg + 1:] = t_values
        
        if valid_func(child, bmi):
            return child
    
    return parent1

def mutate_func(parent: np.ndarray):
    pass


if __name__ == '__main__':
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    y, t, bmi = get_params(df)
    sol1 = init_sol_func(bmi=bmi, n_seg=5)
    sol2 = init_sol_func(bmi=bmi, n_seg=5)
    child = crossover_func(sol1, sol2, bmi)
    print(valid_func(child, bmi))