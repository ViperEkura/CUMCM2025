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
from typing import Dict, List, Tuple

def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    
    wi = Ni / N_total
    gi = Ti - 10
    P = np.sum(wi * gi)
    
    return - P

def run_genetic_algorithm(
    parameters: Dict[str, np.ndarray], 
    n_segments: int, 
    show_progress: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """运行遗传算法优化过程"""
    pop_size = 100
    n_gen = 100
    elitism_ratio = 0.1
    mutation_rate = 0.4
    crossover_rate = 0.8
    
    ga = GeneticAlgorithm(
        pop_size=pop_size,
        n_gen=n_gen,
        init_func=lambda: init_sol_func(parameters, n_segments),
        select_func=lambda pop, fitness: roulette_wheel_select(pop, fitness),
        crossover_func=lambda p1, p2: crossover_func(p1, p2, parameters, crossover_rate),
        mutate_func=lambda ind: mutate_func(ind, parameters, mutation_rate),
        fitness_func=lambda ind: fitness_func(ind, parameters),
        elitism_ratio=elitism_ratio
    )
    
    return ga.run(show_progress)


def get_ga_params(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    week = over_week_df["检测孕周"].values
    
    return {"bmi": bmi, "week": week}

def evaluate_segments(
    parameters: Dict[str, np.ndarray], 
    min_segments: int = 2, 
    max_segments: int = 6, 
    show_results: bool = True
) -> List[Dict]:
    """评估不同分段数量的结果"""
    best_results = []
    print("=" * 50)
    
    for n_segments in range(min_segments, max_segments):
        print(f"Running for n_segments = {n_segments}")
        best_individual, best_fitness_history = run_genetic_algorithm(
            parameters, n_segments, show_progress=False
        )
        
        best_results.append({
            "n_segments": n_segments, 
            "individual": best_individual, 
            "fitness": best_fitness_history[-1]
        })

    if show_results:
        print("-" * 30)
        for result in best_results:
            bmi_divisions = result["individual"]
            n_segments = result["n_segments"]
            fitness_value = result["fitness"]
            ti_values = calcu_Ti(bmi_divisions, parameters)
            
            print(f"Number of segments: {n_segments}")
            print(f"Fitness value: {-fitness_value:.2f}")
            print(f"BMI divisions: {np.around(bmi_divisions, 2)}")
            print(f"TI values: {np.around(ti_values, 2)}")
            print("-" * 30)
    
    return best_results


def error_analysis(df: pd.DataFrame, n_repeats: int = 10, noise_std: float = 0.01):
    """对Y染色体浓度添加扰动后导出结果"""
    all_tis = []
    
    ori_params = get_ga_params(df)
    best_ind, _ = run_genetic_algorithm(ori_params, 5, show_progress=False)
    
    for i in range(n_repeats):
        print(f"Running experiment {i+1}/{n_repeats}")
        df_perturbed = df.copy()
        noise = np.random.normal(0, noise_std, size=len(df))
        df_perturbed["Y染色体浓度"] += noise
        
        params = get_ga_params(df_perturbed)
        ti_values = calcu_Ti(best_ind, params)
        
        all_tis.append(ti_values)
    
    all_tis = np.stack(all_tis, axis=0)
    
    return all_tis



if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    params = get_ga_params(df) 
    best_results = evaluate_segments(params) 
    all_tis = error_analysis(df, n_repeats=10)

    analyze_data(all_tis, "检测孕周阈值误差分析")
