import pandas as pd
import numpy as np
import re

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

def calcu_pregnancy_type(type_info: str):
    trans_dict = { "自然受孕": 1, "IUI（人工授精）": 2, "IVF（试管婴儿）": 3}
    return trans_dict.get(type_info, np.nan)

def calcu_fetus_health(health_info: str):
    trans_dict = {"否": 0, "是": 1}
    return trans_dict.get(health_info, np.nan)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """数据预处理"""
    df.drop(columns=["检测日期", "末次月经"], inplace=True)
    df["检测孕周"] = df["检测孕周"].apply(calcu_week)
    df["IVF妊娠"] = df["IVF妊娠"].apply(calcu_pregnancy_type)
    df["胎儿是否健康"] = df["胎儿是否健康"].apply(calcu_fetus_health)
    
    return df