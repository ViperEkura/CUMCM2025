import pandas as pd
import numpy as np
import re

from scipy import stats
from scipy.stats import shapiro, spearmanr
from typing import Dict, List

def set_seed(seed=3407):
    np.random.seed(seed)

def calcu_week(week_str: str) -> float:
    if pd.isna(week_str) or not isinstance(week_str, str):
        return np.nan

    pattern = r'^(\d+)w(?:\+(\d+))?$'
    matched = re.match(pattern, week_str.strip().lower())
    
    if not matched:
        return np.nan

    weeks = int(matched.group(1))
    days = int(matched.group(2)) if matched.group(2) else 0
    total_weeks = weeks + (days / 7.0)

    return total_weeks

def calcu_pregnancy_type(type_info: str) -> int:
    trans_dict = { "自然受孕": 1, "IUI（人工授精）": 2, "IVF（试管婴儿）": 3}
    return trans_dict.get(type_info, np.nan)

def calcu_fetus_health(health_info: str):
    trans_dict = {"否": 0, "是": 1}
    return trans_dict.get(health_info, np.nan)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理"""
    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    df.drop(columns=["检测日期", "末次月经"], inplace=True)
    df["检测孕周"] = df["检测孕周"].apply(calcu_week)
    df["IVF妊娠"] = df["IVF妊娠"].apply(calcu_pregnancy_type)
    df["胎儿是否健康"] = df["胎儿是否健康"].apply(calcu_fetus_health)
    
    #NA值标记为False，非NA值标记为True
    df["染色体的非整倍体"] = df["染色体的非整倍体"].notna()

    return df
    

def shapiro_test(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    results = []
    for col in columns:
        clean_data = df[col].dropna()
        
        if len(clean_data) < 3:
            results.append({'column': col, 'w': np.nan, 'p': np.nan})
            continue
        
        w, p = shapiro(clean_data)
        results.append({'column': col, 'w': np.around(w, 3), 'p': np.around(p, 3)})
    
    return pd.DataFrame(results)

def spearman_test(df: pd.DataFrame, target_col: str):
    results = []
    for col in df.columns:
        if col != target_col:
            corr, p_value = spearmanr(df[target_col], df[col], nan_policy='omit')
            results.append({
                'column': col,
                'correlation': np.around(corr, 3),
                'p': np.around(p_value, 3)
            })
    
    return pd.DataFrame(results)


def calcu_first_over_week(df: pd.DataFrame, col_name: str, threshold: float):
    def first_over_row(group: pd.DataFrame):
        group = group.sort_values('检测孕周').drop(columns=['孕妇代码'])
        over = group[group[col_name] > threshold]
        if not over.empty:
            return over.iloc[0]
        else:
            return pd.Series([np.nan] * len(group.columns), index=group.columns)

    result = df.groupby('孕妇代码').apply(first_over_row)
    result = result.dropna(subset=['检测孕周'])
    return result


def filter_outliers_iqr(df: pd.DataFrame, feature_cols: List[str], k=1.5):
    mask = pd.Series([True] * len(df), index=df.index)
    
    for col in feature_cols:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask &= col_mask

    return df[mask].reset_index(drop=True)

def custom_statistical_tests(model, X, y, feature_names, coef, intercept):
    def get_display_width(text):
        """计算字符串的显示宽度（中文算2个字符，英文算1个）"""
        width = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width
    
    def pad_to_width(text, width, align='left'):
        """将文本填充到指定显示宽度"""
        text_width = get_display_width(text)
        padding = max(0, width - text_width)
        if align == 'left':
            return text + ' ' * padding
        else:  # right align
            return ' ' * padding + text
    
    def get_significance_star(p_value):
        """获取显著性星号标记"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        elif p_value < 0.1:
            return '.'
        else:
            return ''
    
    y_pred = model.predict(X)
    
    n = len(y)
    k = X.shape[1] 
    
    y_mean = np.mean(y)
    SSR = np.sum((y_pred - y_mean) ** 2)  # 回归平方和
    SSE = np.sum((y - y_pred) ** 2)       # 残差平方和
    SST = np.sum((y - y_mean) ** 2)       # 总平方和
    
    # 1. F检验 - 模型整体显著性
    MSR = SSR / k                         # 回归均方
    MSE = SSE / (n - k - 1)               # 残差均方
    f_statistic = MSR / MSE               # F统计量
    f_pvalue = 1 - stats.f.cdf(f_statistic, k, n - k - 1)  # F检验p值
    
    # 2. R²和调整R²
    r_squared = 1 - (SSE / SST)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
    
    # 计算各列宽度
    name_width = max(max(get_display_width(name) for name in feature_names), 
                    get_display_width('截距'), 
                    get_display_width('变量')) + 2
    
    # 输出F检验结果
    separator = "=" * (name_width + 55)
    print(separator)
    print("F检验 - 模型整体显著性")
    print(separator)
    print(f"{pad_to_width('F统计量:', name_width)} F({k},{n-k-1}) = {f_statistic:.4f}")
    print(f"{pad_to_width('P值:', name_width)} {f_pvalue:.4f}")
    print(f"{pad_to_width('模型显著性:', name_width)} {'显著' if f_pvalue < 0.05 else '不显著'}")
    print(f"{pad_to_width('R²:', name_width)} {r_squared:.4f}")
    print(f"{pad_to_width('调整R²:', name_width)} {adjusted_r_squared:.4f}")
    
    # 3. t检验
    try:
        # 计算系数的标准误（需要矩阵运算）
        X_design = np.column_stack([np.ones(n), X])  # 添加截距列
        covariance_matrix = MSE * np.linalg.inv(X_design.T @ X_design)
        standard_errors = np.sqrt(np.diag(covariance_matrix))
        
        print("\n" + separator)
        print("t检验 - 各个变量显著性")
        print(separator)
        
        # 表头
        header = (pad_to_width('变量', name_width) + 
                 pad_to_width('系数', 10, 'right') +
                 pad_to_width('标准误', 10, 'right') +
                 pad_to_width('t值', 10, 'right') +
                 pad_to_width('P值', 10, 'right') +
                 pad_to_width('显著性', 8, 'right'))
        print(header)
        print("-" * (name_width + 55))
        
        # 截距项的t检验
        t_value_intercept = intercept / standard_errors[0]
        p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_value_intercept), n - k - 1))
        
        sig = get_significance_star(p_value_intercept)
        row = (pad_to_width('截距', name_width) +
               pad_to_width(f"{intercept:.4f}", 10, 'right') +
               pad_to_width(f"{standard_errors[0]:.4f}", 10, 'right') +
               pad_to_width(f"{t_value_intercept:.4f}", 10, 'right') +
               pad_to_width(f"{p_value_intercept:.4f}", 10, 'right') +
               pad_to_width(sig, 8, 'right'))
        print(row)
        
        # 各个特征的t检验
        for i, name in enumerate(feature_names):
            t_value = coef[i] / standard_errors[i+1]
            p_value = 2 * (1 - stats.t.cdf(abs(t_value), n - k - 1))
            
            sig = get_significance_star(p_value)
            row = (pad_to_width(name, name_width) +
                   pad_to_width(f"{coef[i]:.4f}", 10, 'right') +
                   pad_to_width(f"{standard_errors[i+1]:.4f}", 10, 'right') +
                   pad_to_width(f"{t_value:.4f}", 10, 'right') +
                   pad_to_width(f"{p_value:.4f}", 10, 'right') +
                   pad_to_width(sig, 8, 'right'))
            print(row)
        
    except Exception as e:
        print(f"警告: 无法计算t检验: {e}")
        print("可能原因: 设计矩阵不满秩或存在多重共线性")
    
    print("-" * (name_width + 55))
    print("显著性水平: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
    print("=" * (name_width + 55))
    
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

def sensitivity_analysis(
    df:pd.DataFrame, 
    n_segments:int, 
    run_ga_func, 
    get_params_func, 
    calc_ti_func, 
    n_repeats=5, 
    change_percent=0.01
):
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

def sensitivity_summary(results: Dict, n_segments: int):
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
