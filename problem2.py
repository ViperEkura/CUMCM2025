import os
import numpy as np
import pandas as pd
from scipy import stats
from utils.ga import run_genetic_algorithm, calcu_Ti
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import Dict

plot_save_path = 'analyze_plot'
table_save_path = 'analyze_table'
os.makedirs(plot_save_path, exist_ok=True)
os.makedirs(table_save_path, exist_ok=True)

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
        best_ind, best_fitnesses = run_genetic_algorithm(params, n_seg, show_progress=True)
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
    """对Y染色体浓度添加扰动后导出结果"""
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
        
        all_inds.append(best_ind)
        all_tis.append(ti_values)
    
    all_inds = np.stack(all_inds, axis=0)
    all_tis = np.stack(all_tis, axis=0)
    
    return all_inds, all_tis

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)
    
    margin_of_error = t_critical * std_err
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper

def analyze_data(data: np.ndarray, data_type: str ="", start_index: int = 0) -> None:
    n_params = data.shape[1]
    
    print(f"\n{data_type}统计结果:")
    print("=" * 90)
    print("序号 |    均值    |   标准差   | 变异系数(%) |    95%置信区间    |     区间宽度")
    print("-" * 90)
    
    for i in range(n_params):
        param_data = data[:, i]
        mean = np.mean(param_data)
        std = np.std(param_data)
        cv = (std / mean) * 100 if mean != 0 else 0

        _, ci_lower, ci_upper = calculate_confidence_interval(param_data)
        interval_width = ci_upper - ci_lower
        
        index = start_index + i
        print(f"{index:4d} | {mean:10.4f} | {std:10.4f} | {cv:10.2f} | [{ci_lower:7.4f}, {ci_upper:7.4f}] | {interval_width:10.4f}")


if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    
    best_results = show_segments(df)
    all_inds, all_tis = error_analysis(df, n_repeats=10) # 根据计算资源调整

    analyze_data(all_inds, "BMI分段点误差分析")
    analyze_data(all_tis, "检测孕周阈值误差分析")
