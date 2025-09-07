import os  
os.environ["OMP_NUM_THREADS"] = "1"  

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.ga import (
    GeneticAlgorithm, 
    calcu_Ni, 
    calcu_Ti,
    calcu_R_CA,
    calcu_R_IVF,
    calcu_R_GC,
    init_sol_func, 
    roulette_wheel_select, 
    crossover_func, 
    mutate_func
)
from utils.data_util import preprocess, calcu_first_over_week, set_seed
from typing import Dict, List, Tuple


def get_ga_params(df: pd.DataFrame, n_clusters=3) -> Dict[str, Dict[str, np.ndarray]]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(over_week_df[["年龄", "孕妇BMI"]])
    over_week_df["clusters"] = kmeans.labels_
    
    cluster_groups = {}
    for cluster_id in range(n_clusters):
        cluster_data = over_week_df[over_week_df["clusters"] == cluster_id]
        
        cluster_params = {
            "bmi": cluster_data["孕妇BMI"].values,
            "week": cluster_data["检测孕周"].values,
            "ivf": cluster_data["IVF妊娠"].values == 3, # IVF(试管婴儿)
            "ca": cluster_data["染色体的非整倍体"].values,
            "gc": cluster_data["GC含量"].values
        }
        
        cluster_groups[f"cluster_{cluster_id}"] = cluster_params
    
    return cluster_groups

def fitness_func(
    ind: np.ndarray, 
    params: Dict[str, np.ndarray],
    alpha=10,
    beta=10,
    gamma=10,
    zeta=10,
):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    wi = Ni / N_total
    
    h = alpha * Ti
    R_ivf = beta * calcu_R_IVF(ind, params)
    R_ca = gamma * calcu_R_CA(ind, params)
    R_gc = zeta * calcu_R_GC(ind, params)
    Z = np.sum(wi * (h + R_ivf + R_ca + R_gc))
    
    return -Z

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


def show_multi_segments(
    cluster_params: Dict[str, Dict[str, np.ndarray]], 
) -> Dict[str, dict]:
    cls0 = cluster_params["cluster_0"]
    cls1 = cluster_params["cluster_1"]
    cls2 = cluster_params["cluster_2"]
    
    evaluate_segments(cls0, 2, 3)
    evaluate_segments(cls1, 2, 3)
    evaluate_segments(cls2, 2, 5)


if __name__ == "__main__":
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    multi_params = get_ga_params(df)
    show_multi_segments(multi_params)

