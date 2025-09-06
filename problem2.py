import os
import numpy as np
import pandas as pd
from utils.ga import (
    GeneticAlgorithm, 
    calcu_Ni, 
    calcu_Ti, 
    init_sol_func, 
    roulette_wheel_select, 
    crossover_func, 
    mutate_func
)
from utils.data_uitl import analyze_data, preprocess, calcu_first_over_week, set_seed
from typing import Dict

plot_save_path = 'analyze_plot'
table_save_path = 'analyze_table'
os.makedirs(plot_save_path, exist_ok=True)
os.makedirs(table_save_path, exist_ok=True)
    
    
def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    
    wi = Ni / N_total
    gi = Ti - 10
    P = np.sum(wi * gi)
    
    return - P

def run_genetic_algorithm(params: Dict[str, np.ndarray], n_seg: int, show_progress: bool):
    pop_size = 100
    n_gen = 100
    elitism_ratio = 0.1
    mutate_rate = 0.4
    crossover_rate = 0.8
    fitness_fn = lambda ind: fitness_func(ind, params)
    init_fn = lambda: init_sol_func(params, n_seg)
    select_fn = lambda pop, fitness: roulette_wheel_select(pop, fitness)
    crossover_fn = lambda parent1, parent2: crossover_func(parent1, parent2, params, crossover_rate)
    mutate_fn = lambda parent: mutate_func(parent, params, mutate_rate)

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
    
    return ga.run(show_progress)

def get_ga_params(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    week = over_week_df["检测孕周"].values
    
    return {"bmi": bmi, "week": week}

def show_segments(params: Dict[str, np.ndarray], n_start=2, n_end=6, show_res: bool=True):
    best_results = []
    print("="*50)
    for n_seg in range(n_start, n_end):
        print(f"Running for n_seg = {n_seg}")
        best_ind, best_fitnesses = run_genetic_algorithm(params, n_seg, show_progress=False)
        best_results.append({"n_seg": n_seg, "ind": best_ind, "fitnesses": best_fitnesses[-1]})

    if show_res:
        for res in best_results:
            bmi_div = res["ind"]
            n_seg = res["n_seg"]
            fitness = res["fitnesses"]
            ti = calcu_Ti(bmi_div, params)
            
            print(f"n_seg: {n_seg}")
            print(f"fitness: {-fitness:.2f}")
            print(f"b: {np.around(bmi_div, 2)}")
            print(f"t: {np.around(ti, 2)}")
    
    return best_results


def error_analysis(df: pd.DataFrame, n_repeats: int = 10, noise_std: float = 0.01):
    """对Y染色体浓度添加扰动后导出结果"""
    all_inds = []
    all_tis = []
    
    for i in range(n_repeats):
        print(f"Running experiment {i+1}/{n_repeats}")
        df_perturbed = df.copy()
        noise = np.random.normal(0, noise_std, size=len(df))
        df_perturbed["Y染色体浓度"] += noise
        
        params = get_ga_params(df_perturbed)
        best_ind, _ = run_genetic_algorithm(params, 5, show_progress=False)
        ti_values = calcu_Ti(best_ind, params)
        
        all_inds.append(best_ind)
        all_tis.append(ti_values)
    
    all_inds = np.stack(all_inds, axis=0)
    all_tis = np.stack(all_tis, axis=0)
    
    return all_inds, all_tis



if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    
    best_results = show_segments(df)
    all_inds, all_tis = error_analysis(df, n_repeats=10) # 根据计算资源调整

    analyze_data(all_inds, "BMI分段点误差分析")
    analyze_data(all_tis, "检测孕周阈值误差分析")
