import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from utils.data_uitl import preprocess, calcu_first_over_week
from utils.ga import calcu_Ti, calcu_Ni
from typing import Dict


def get_init_params(df: pd.DataFrame, n_clusters) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(over_week_df[["年龄", "孕妇BMI"]])
    over_week_df["clusters"] = kmeans.labels_
    
    params = {
        "cls": over_week_df["clusters"].values, 
        "bmi": over_week_df["孕妇BMI"].values,
        "week": over_week_df["检测孕周"].values,
        "ivf": over_week_df["IVF妊娠"].values,
        "t13": over_week_df["T13"].values,
        "t18": over_week_df["T18"].values,
        "t21": over_week_df["T21"].values
    }
    
    return params


def R_ivf():
    pass

def R_abnromal():
    pass


def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    wi = Ni / N_total
    
    
    return None


def valid_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    pass


if __name__ == "__main__":
    n_clusters = 3
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    params = get_init_params(df, n_clusters)
    
    print(params)