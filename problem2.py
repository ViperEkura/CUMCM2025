import numpy as np
import pandas as pd
from utils.ga import run_genetic_algorithm, calcu_Ti
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import Dict, List

def set_seed(seed=3407):
    np.random.seed(seed)

def get_ga_params(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    week = over_week_df["检测孕周"].values
    
    return {"bmi": bmi, "week": week}

def show_segments(df: pd.DataFrame, n_start=2, n_end=6, show_res: bool=True, show_progress: bool=True):
    params = get_ga_params(df)
    best_results = []
    for n_seg in range(n_start, n_end):
        best_ind, best_fitnesses = run_genetic_algorithm(params, n_seg, show_progress)
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
    """对Y染色体浓度添加扰动后运行遗传算法，分析节点(bmi_div)和Ti的稳定性"""
    result_groups = []
    
    for i in range(n_repeats):
        print(f"Running perturbation experiment {i+1}/{n_repeats}")
        df_perturbed = df.copy()
        noise = np.random.normal(0, noise_std, size=len(df))
        df_perturbed["Y染色体浓度"] += noise
        
        params = get_ga_params(df_perturbed)
        # 根据之前的数据选择第五个
        results = run_genetic_algorithm(params, 5, show_progress=False)
        result_groups.append(results)
    



if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    
    show_segments(df)
    error_analysis(df)