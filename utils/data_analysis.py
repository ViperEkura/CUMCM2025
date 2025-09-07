from matplotlib import pyplot as plt
import numpy as np


def sensitivity_analysis(df, n_segments, run_ga_func, get_params_func, calc_ti_func, n_repeats=5, change_percent=0.01):

    original_params = get_params_func(df)
    
    best_ind, _ = run_ga_func(original_params, n_segments, show_progress=False)
    original_ti = calc_ti_func(best_ind, original_params)
    
    results = {
        'original': {'ti': original_ti, 'bmi_divisions': best_ind},
        'positive_perturb': {'ti': [], 'bmi_divisions': []},
        'negative_perturb': {'ti': [], 'bmi_divisions': []}
    }
    
    for i in range(n_repeats):
        print(f"Running sensitivity analysis {i+1}/{n_repeats}")
        
        # 正向扰动 (+1%)
        df_positive = df.copy()
        df_positive["Y染色体浓度"] = df_positive["Y染色体浓度"] * (1 + change_percent)
        positive_params = get_params_func(df_positive)
        positive_ind, _ = run_ga_func(positive_params, n_segments, show_progress=False)
        positive_ti = calc_ti_func(positive_ind, positive_params)
        
        results['positive_perturb']['ti'].append(positive_ti)
        results['positive_perturb']['bmi_divisions'].append(positive_ind)
        
        # 负向扰动 (-1%)
        df_negative = df.copy()
        df_negative["Y染色体浓度"] = df_negative["Y染色体浓度"] * (1 - change_percent)
        negative_params = get_params_func(df_negative)
        negative_ind, _ = run_ga_func(negative_params, n_segments, show_progress=False)
        negative_ti = calc_ti_func(negative_ind, negative_params)
        
        results['negative_perturb']['ti'].append(negative_ti)
        results['negative_perturb']['bmi_divisions'].append(negative_ind)
    
    # 计算平均结果
    for key in ['positive_perturb', 'negative_perturb']:
        results[key]['ti_mean'] = np.mean(results[key]['ti'], axis=0)
        results[key]['ti_std'] = np.std(results[key]['ti'], axis=0)
        results[key]['bmi_mean'] = np.mean(results[key]['bmi_divisions'], axis=0)
        results[key]['bmi_std'] = np.std(results[key]['bmi_divisions'], axis=0)
    
    return results


def plot_sensitivity_results(results, n_segments=5, save_path='sensitivity_analysis.png'):
    """
    绘制灵敏度分析结果
    
    参数:
    results: 灵敏度分析结果
    n_segments: 分段数量
    save_path: 保存图像的路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # TI值变化
    segments = range(1, n_segments+1)
    ax1.errorbar(segments, results['positive_perturb']['ti_mean'], 
                yerr=results['positive_perturb']['ti_std'], 
                fmt='o-', label='+1% Y染色体浓度', capsize=5)
    ax1.errorbar(segments, results['original']['ti'], 
                yerr=0, fmt='s-', label='原始', capsize=5)
    ax1.errorbar(segments, results['negative_perturb']['ti_mean'], 
                yerr=results['negative_perturb']['ti_std'], 
                fmt='o-', label='-1% Y染色体浓度', capsize=5)
    ax1.set_xlabel('分段')
    ax1.set_ylabel('TI值')
    ax1.set_title('Y染色体浓度变化对TI值的影响')
    ax1.legend()
    ax1.grid(True)
    

def sensitivity_summary(results, n_segments=5):
    """
    打印灵敏度分析摘要
    
    参数:
    results: 灵敏度分析结果
    n_segments: 分段数量
    """
    print("="*50)
    print("灵敏度分析摘要")
    print("="*50)
    
    # TI值变化
    print("\nTI值变化:")
    print(f"{'分段':<6} {'原始':<8} {'+1%':<8} {'变化(%)':<8} {'-1%':<8} {'变化(%)':<8}")
    for i in range(n_segments):
        orig_ti = results['original']['ti'][i]
        pos_ti = results['positive_perturb']['ti_mean'][i]
        neg_ti = results['negative_perturb']['ti_mean'][i]
        
        pos_change = (pos_ti - orig_ti) / orig_ti * 100
        neg_change = (neg_ti - orig_ti) / orig_ti * 100
        
        print(f"{i+1:<6} {orig_ti:<8.2f} {pos_ti:<8.2f} {pos_change:<8.2f}% {neg_ti:<8.2f} {neg_change:<8.2f}%")
    
    # BMI分段点变化
    n_bmi_points = len(results['original']['bmi_divisions'])
    print(f"\nBMI分段点变化 (共{n_bmi_points}个点):")
    print(f"{'点':<4} {'原始':<8} {'+1%':<8} {'变化(%)':<8} {'-1%':<8} {'变化(%)':<8}")
    for i in range(n_bmi_points):
        orig_bmi = results['original']['bmi_divisions'][i]
        pos_bmi = results['positive_perturb']['bmi_mean'][i]
        neg_bmi = results['negative_perturb']['bmi_mean'][i]
        
        pos_change = (pos_bmi - orig_bmi) / orig_bmi * 100
        neg_change = (neg_bmi - orig_bmi) / orig_bmi * 100
        
        print(f"{i+1:<4} {orig_bmi:<8.2f} {pos_bmi:<8.2f} {pos_change:<8.2f}% {neg_bmi:<8.2f} {neg_change:<8.2f}%")