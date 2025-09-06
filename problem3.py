import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.data_uitl import preprocess, calcu_first_over_week
from utils.ga import calcu_Ti
from typing import Dict

def Ra():
    pass

def Rbcd():
    pass


def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ti(ind, params)
    Ti = calcu_Ti(ind, params)
    
    wi = Ni / N_total
    gi = Ti - 10
    P = np.sum(wi * gi)
    
    return - P


def valid_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    pass


def get_init_params():
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    return over_week_df


def get_center_and_label(df: pd.DataFrame):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df)
    return kmeans.cluster_centers_, kmeans.labels_


if __name__ == "__main__":
    
    df = get_init_params()
    center, label = get_center_and_label(df)

