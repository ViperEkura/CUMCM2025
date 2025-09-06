import numpy as np
import pandas as pd
from utils.ga import GeneticAlgorithm
from utils.data_uitl import preprocess, calcu_first_over_week
from typing import Dict, List


def set_seed(seed=3407):
    np.random.seed(seed)

def get_params(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    over_week_df = calcu_first_over_week(df, "Y染色体浓度", 0.04)
    over_week_df = over_week_df.sort_values("孕妇BMI")
    bmi = over_week_df["孕妇BMI"].values
    week = over_week_df["检测孕周"].values
    
    return {"bmi": bmi, "week": week}

def calcu_Ni(ind: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
    bmi = params["bmi"]
    n_seg = np.size(ind) - 1
    Ni = np.zeros(n_seg)
    
    for i in range(n_seg):
        start_bmi = ind[i]
        end_bmi = ind[i + 1]
        mask = (bmi >= start_bmi) & (bmi < end_bmi)
        Ni[i] = np.sum(mask)
    
    return Ni

def calcu_Ti(ind: np.ndarray, params: Dict[str, np.ndarray], percent: float=90):
    bmi = params["bmi"]
    week = params["week"]
    n_seg = np.size(ind) - 1
    Ti = np.zeros(n_seg)
    
    for i in range(n_seg):
        start_bmi = ind[i]
        end_bmi = ind[i + 1]
        mask = (bmi >= start_bmi) & (bmi < end_bmi)
        week_in_interval = week[mask]
        Ti[i] = np.percentile(week_in_interval, percent)
    
    return Ti

def valid_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    Ni = calcu_Ni(ind, params)
    if np.any(Ni <= 25):
        return False
    
    Ti = calcu_Ti(ind, params)
    if np.any(Ti < 10) or np.any(Ti > 25):
        return False
    
    return True

def init_sol_func(params:Dict[str, np.ndarray], n_seg: int):
    bmi = params["bmi"]
    
    bmi_min, bmi_max = np.min(bmi), np.max(bmi)
    bmi_div = np.random.uniform(bmi_min, bmi_max, (n_seg + 1))
    bmi_div = np.sort(bmi_div, axis=-1)
    bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)
    
    ind = bmi_div
    
    while not valid_func(ind, params):
        bmi_div = np.random.uniform(bmi_min, bmi_max, (n_seg + 1))
        bmi_div = np.sort(bmi_div, axis=-1)
        bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)

        ind = bmi_div
    
    return ind

def crossover_func(
    parent1: np.ndarray, 
    parent2: np.ndarray, 
    params:Dict[str, np.ndarray], 
    crossover_ratio: float = 0.8, 
    max_attempts: int = 2000
) -> np.ndarray:
    """多点交叉"""
    n_seg = np.size(parent1) - 1
    bmi_min, bmi_max = np.min(params["bmi"]), np.max(params["bmi"])
    
    for _ in range(max_attempts):
        child = parent1.copy()
        
        crossover_points = np.random.choice(range(1, n_seg), 
                                          size=int(crossover_ratio * n_seg), 
                                          replace=False)
        
        for i in crossover_points:
            child[i] = parent2[i]

        child = np.sort(child)
        child[0] = bmi_min
        child[-1] = bmi_max
        
        if valid_func(child, params):
            return child

    fitness1 = fitness_func(parent1, params)
    fitness2 = fitness_func(parent2, params)
    return parent1 if fitness1 > fitness2 else parent2

def mutate_func(
    parent: np.ndarray,
    params: Dict[str, np.ndarray], 
    mutation_rate: float = 0.3,
    max_attempts: int = 2000
):
    """均匀变异"""
    n_seg = np.size(parent) - 1
    bmi_min, bmi_max = np.min(params["bmi"]), np.max(params["bmi"])
    
    for _ in range(max_attempts):
        child = parent.copy()
        
        mutation_points = np.random.choice(range(1, n_seg), 
                                         size=int(mutation_rate * n_seg), 
                                         replace=False)
        
        for i in mutation_points:
            left_bound = child[i-1] if i > 0 else bmi_min
            right_bound = child[i+1] if i < n_seg else bmi_max
            
            new_value = np.random.uniform(left_bound, right_bound)
            child[i] = new_value
        
        child = np.sort(child)
        child[0] = bmi_min
        child[-1] = bmi_max
        
        if valid_func(child, params):
            return child
    
    return parent

def roulette_wheel_select(
    population: List[np.ndarray], 
    fitness_values:List[float], 
    num_selected: int=2
):
    max_fitness = np.max(fitness_values)
    adjusted_fitness = max_fitness - np.array(fitness_values) + 1e-6
    
    total_fitness = np.sum(adjusted_fitness)
    selection_probs = adjusted_fitness / total_fitness
    
    cumulative_probs = np.cumsum(selection_probs)
    selected_indices = []
    
    for _ in range(num_selected):
        r = np.random.random()
        for i, cum_prob in enumerate(cumulative_probs):
            if r <= cum_prob:
                selected_indices.append(i)
                break

    return [population[i] for i in selected_indices]

def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    
    wi = Ni / N_total
    gi = Ti - 10
    P = np.sum(wi * gi)
    
    return - P

def run_genetic_algorithm(params: Dict[str, np.ndarray]):
    pop_size = 100
    n_gen = 100
    elitism_ratio = 0.05
    mutate_rate = 0.1
    crossover_rate = 0.5
    fitness_fn = lambda ind: fitness_func(ind, params)
    
    best_results = []
    
    for n_seg in range(2, 7):
        print(f"Running for n_seg = {n_seg}")
        print("="*50)
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
        
        best_ind, best_fitnesses = ga.run()
        best_results.append({"n_seg": n_seg, "ind": best_ind, "fitnesses": best_fitnesses})
    
    return best_results


def export_results(results: List[Dict[str, np.ndarray]], params: Dict[str, np.ndarray], filename: str):
    for res in results:
        bmi_div = res["ind"]
        n_seg = res["n_seg"]
        ti = calcu_Ti(bmi_div, params)
        
        print(f"n_seg: {n_seg}")
        print(f"b: {bmi_div}")
        print(f"t: {ti}")


def error_analysis(y_true, y_pred):
    pass


if __name__ == '__main__':
    set_seed()
    df = preprocess(pd.read_excel('附件.xlsx', sheet_name=0))
    params = get_params(df)
    results = run_genetic_algorithm(params)
    export_results(results, params, "result.txt")