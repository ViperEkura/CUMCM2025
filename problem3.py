import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.ga import (
    GeneticAlgorithm, 
    calcu_Ni, 
    calcu_Ti,
    calcu_R_CA,
    calcu_R_IVF,
    init_sol_func, 
    roulette_wheel_select, 
    crossover_func, 
    mutate_func
)
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import Dict


def get_init_params(df: pd.DataFrame, n_clusters=3) -> Dict[str, Dict[str, np.ndarray]]:
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
            "ca": cluster_data["染色体的非整倍体"].values
        }
        
        cluster_groups[f"cluster_{cluster_id}"] = cluster_params
    
    return cluster_groups

def fitness_func(
    ind: np.ndarray, 
    params: Dict[str, np.ndarray],
    alpha=10,
    beta=10,
    gamma=10,
):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    wi = Ni / N_total
    
    h = alpha * Ti
    R_ivf = beta * calcu_R_IVF(ind, params)
    R_ca = gamma * calcu_R_CA(ind, params)
    Z = np.sum(wi * (h + R_ivf + R_ca))
    
    return -Z

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
       
    print("="*50)
    return best_results


def run_multi_cluster_ga(
    cluster_params: Dict[str, Dict[str, np.ndarray]], 
) -> Dict[str, dict]:
    cls0 = cluster_params["cluster_0"]
    cls1 = cluster_params["cluster_1"]
    cls2 = cluster_params["cluster_2"]
    
    res0 = show_segments(cls0, 2, 3)
    res1 = show_segments(cls1, 2, 3)
    res2 = show_segments(cls2, 2, 5)
        



if __name__ == "__main__":
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    params = get_init_params(df)
    run_multi_cluster_ga(params)
