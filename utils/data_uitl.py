import pandas as pd
import numpy as np
import re

from scipy.stats import shapiro, spearmanr
from typing import List


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
    df = df[(df["GC含量"] >= 0.4) & (df["GC含量"] <= 0.6)]
    
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
