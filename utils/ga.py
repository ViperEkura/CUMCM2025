import numpy as np
from typing import Callable, Dict, List, Tuple


class GeneticAlgorithm:
    def __init__(
        self, 
        pop_size: int, 
        n_gen: int,
        init_func: Callable[..., np.ndarray],
        select_func: Callable[[List[np.ndarray], List[float]], Tuple[np.ndarray, np.ndarray]],
        crossover_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mutate_func: Callable[[np.ndarray], np.ndarray],
        fitness_func: Callable[[np.ndarray], float],
        elitism_ratio: float = 0.1
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.init_func = init_func
        self.select_func = select_func
        self.crossover_func = crossover_func
        self.mutate_func = mutate_func
        self.fitness_func = fitness_func
        self.elitism_ratio = elitism_ratio
        self.n_elite = max(1, int(pop_size * elitism_ratio))
    
    def run(self, show_progress=True) -> Tuple[np.ndarray, List[float]]:
        population = [self.init_func() for _ in range(self.pop_size)]
        best_fitness_history = []
        
        for gen in range(self.n_gen):
            fitnesses = [self.fitness_func(individual) for individual in population]
            
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            
            # 使用字典来去重，键为个体的字符串表示
            unique_individuals = {}
            for i, ind in enumerate(population):
                ind_key = tuple(ind) 
                if ind_key not in unique_individuals or fitnesses[i] > fitnesses[unique_individuals[ind_key]]:
                    unique_individuals[ind_key] = i
            
            sorted_unique_indices = sorted(unique_individuals.values(), 
                                        key=lambda i: fitnesses[i], 
                                        reverse=True)
            
            elite_indices = sorted_unique_indices[:self.n_elite]
            elites = [population[i] for i in elite_indices]

            new_population = elites.copy()
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_func(population, fitnesses)
                child = self.crossover_func(parent1, parent2)
                child = self.mutate_func(child)
                new_population.append(child)
            
            population = new_population
            
            # 输出多样性信息
            if show_progress:
                unique_count = len(set(tuple(ind) for ind in population))
                print(f"Generation {gen}: Best Fitness = {best_fitness}, Diversity = {unique_count}/{self.pop_size}")
        
        final_fitnesses = [self.fitness_func(individual) for individual in population]
        best_individual = population[np.argmax(final_fitnesses)]
        
        return best_individual, best_fitness_history
    

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
    bmi_div = np.zeros(n_seg + 1)
    bmi_div[0], bmi_div[-1] = np.min(bmi), np.max(bmi)
    bmi_div[1:-1] = np.random.uniform(bmi_min, bmi_max, (n_seg - 1,))
    bmi_div = np.sort(bmi_div, axis=-1)
    
    while not valid_func(bmi_div, params):
        bmi_div[1:-1] = np.random.uniform(bmi_min, bmi_max, (n_seg - 1,))
        bmi_div = np.sort(bmi_div, axis=-1)
    
    return bmi_div

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
    min_fitness = np.min(fitness_values)
    adjusted_fitness = np.array(fitness_values) - min_fitness + 1e-6
    
    total_fitness = np.sum(adjusted_fitness)
    selection_probs = adjusted_fitness / total_fitness
    indices = np.random.choice(len(population), size=num_selected, p=selection_probs)
    
    return [population[i] for i in indices]

def fitness_func(ind: np.ndarray, params: Dict[str, np.ndarray]):
    N_total = np.size(params["bmi"])
    Ni = calcu_Ni(ind, params)
    Ti = calcu_Ti(ind, params)
    
    wi = Ni / N_total
    gi = Ti - 10
    P = np.sum(wi * gi)
    
    return - P

def run_genetic_algorithm(params: Dict[str, np.ndarray], n_seg: int, show_progress: bool):
    pop_size = 100
    n_gen = 100
    elitism_ratio = 0.1
    mutate_rate = 0.4
    crossover_rate = 0.8
    fitness_fn = lambda ind: fitness_func(ind, params)
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
    
    return ga.run(show_progress)