import numpy as np
import pandas as pd
from utils.plot_util import plot_confidence_intervals
from utils.ga import run_genetic_algorithm, calcu_Ti
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import Dict

def set_seed(seed=3407):
    np.random.seed(seed)

def get_ga_params(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    week = over_week_df["检测孕周"].values
    
    return {"bmi": bmi, "week": week}

def show_segments(df: pd.DataFrame, n_start=2, n_end=6, show_res: bool=True):
    params = get_ga_params(df)
    best_results = []
    for n_seg in range(n_start, n_end):
        print(f"Running for n_seg = {n_seg}")
        best_ind, best_fitnesses = run_genetic_algorithm(params, n_seg, False)
        best_results.append({"n_seg": n_seg, "ind": best_ind, "fitnesses": best_fitnesses[-1]})

    print("="*50)
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
    """简化版误差分析：对Y染色体浓度添加扰动后分析稳定性"""
    all_inds = []
    all_tis = []
    
    original_params = get_ga_params(df)
    
    for i in range(n_repeats):
        print(f"Running experiment {i+1}/{n_repeats}")
        df_perturbed = df.copy()
        noise = np.random.normal(0, noise_std, size=len(df))
        df_perturbed["Y染色体浓度"] += noise
        
        params = get_ga_params(df_perturbed)
        best_ind, _ = run_genetic_algorithm(params, 5, show_progress=False)
        ti_values = calcu_Ti(best_ind, original_params)
        
        all_inds.append(best_ind[1:-1])
        all_tis.append(ti_values)
    
    ind_array = np.array(all_inds)
    ti_array = np.array(all_tis)
    
    print("\n误差分析结果 (n_seg=5):")
    print("=" * 40)
    
    print("\nBMI分段点统计:")
    print("序号 | 均值   | 标准差 | 变异系数(%)")
    print("-" * 35)
    for i in range(ind_array.shape[1]):
        mean = np.mean(ind_array[:, i])
        std = np.std(ind_array[:, i])
        cv = (std / mean) * 100
        print(f"{i:4d} | {mean:6.2f} | {std:6.3f} | {cv:8.2f}")
    
    print("\nTi值统计:")
    print("分段 | 均值   | 标准差 | 变异系数(%)")
    print("-" * 35)
    for i in range(ti_array.shape[1]):
        mean = np.mean(ti_array[:, i])
        std = np.std(ti_array[:, i])
        cv = (std / mean) * 100
        print(f"{i+1:4d} | {mean:6.2f} | {std:6.3f} | {cv:8.2f}")
        
    return all_inds, all_tis


if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    
    show_segments(df)
    error_analysis(df)