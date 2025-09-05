import numpy as np
import pandas as pd
from utils.ga import GeneticAlgorithm
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import List

def get_params(df: pd.DataFrame):
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    return bmi

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

def mutate_func(parent: np.ndarray, bmi: np.ndarray, mutation_rate: float = 0.3, max_attempts: int = 20):
    n_seg = (len(parent) - 1) // 2
    child = np.copy(parent)
    
    if np.random.random() > mutation_rate:
        return child
    
    for _ in range(max_attempts):

        if np.random.random() < 0.5:  
            # 变异BMI分界点
            if n_seg > 1:
                idx = np.random.randint(1, n_seg)
                low_bound = child[idx-1]
                high_bound = child[idx+1]
                new_value = np.random.uniform(low_bound, high_bound)
                child[idx] = new_value
        else:  
            # 变异t值
            idx = np.random.randint(0, n_seg)
            new_t = np.random.uniform(10, 25)
            child[n_seg+1+idx] = new_t
        
        # 确保边界条件
        child[0] = np.min(bmi)
        child[n_seg] = np.max(bmi)
        
        # 检查可行性
        if valid_func(child, bmi):
            return child

    return parent

def roulette_wheel_select(population: List[np.ndarray], fitness_values:List[float], num_selected: int=2):
    inverted_fitness = 1.0 / (np.array(fitness_values) + 1e-6)

    total_fitness = np.sum(inverted_fitness)
    selection_probs = inverted_fitness / total_fitness
    
    cumulative_probs = np.cumsum(selection_probs)
    selected_indices = []
    
    for _ in range(num_selected):
        r = np.random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if r <= cum_prob:
                selected_indices.append(i)
                break
    

    return [population[i] for i in selected_indices]

def fitness_func(ind: np.ndarray, bmi: np.ndarray):
    n_seg = (len(ind) - 1) // 2
    t = ind[n_seg + 1:]
    
    Ni = calcu_Ni(ind, bmi)
    N_total = len(bmi)
    
    wi = Ni / N_total
    
    g = np.zeros(n_seg)
    for i in range(n_seg):
        if t[i] <= 12:
            g[i] = 1
        else:
            g[i] = 2

    P = 2 - np.sum(wi * g)
    
    return P

def run_genetic_algorithm(bmi: np.ndarray):
    pop_size = 100
    n_gen = 100
    elitism_ratio = 0.2
    mutate_rate = 0.5
    fitness_fn = lambda ind: fitness_func(ind, bmi)
    
    for n_seg in range(2, 6):
        print(f"Running for n_seg = {n_seg}")
        print("="*50)
        init_fn = lambda: init_sol_func(bmi, n_seg)
        select_fn = lambda pop, fitness: roulette_wheel_select(pop, fitness)
        crossover_fn = lambda parent1, parent2: crossover_func(parent1, parent2, bmi)
        mutate_fn = lambda parent: mutate_func(parent, bmi, mutate_rate)
        
        ga = GeneticAlgorithm(
            pop_size, 
            n_gen, 
            init_fn, 
            select_fn, 
            crossover_fn, 
            mutate_fn, 
            fitness_fn, 
            elitism_ratio
        )
        
        ga.run()


if __name__ == '__main__':
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    bmi = get_params(df)
    run_genetic_algorithm(bmi)