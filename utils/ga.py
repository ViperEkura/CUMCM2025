import numpy as np
from typing import Callable, List, Tuple


class GeneticAlgorithm:
    def __init__(
        self, 
        pop_size: int, 
        n_gen: int,
        init_func: Callable[..., np.ndarray],
        select_func: Callable[[List[np.ndarray], List[float]], Tuple[np.ndarray, np.ndarray]],
        crossover_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        mutation_func: Callable[[np.ndarray], np.ndarray],
        fitness_func: Callable[[np.ndarray], float],
        elitism_ratio: float = 0.1
    ):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.init_func = init_func
        self.select_func = select_func
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.fitness_func = fitness_func
        self.elitism_ratio = elitism_ratio
        self.n_elite = max(1, int(pop_size * elitism_ratio))
    
    def run(self) -> Tuple[np.ndarray, List[float]]:
        population = [self.init_func() for _ in range(self.pop_size)]
        best_fitness_history = []
        
        for _ in range(self.n_gen):
            fitnesses = [self.fitness_func(individual) for individual in population]
            
            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            
            elite_indices = np.argsort(fitnesses)[-self.n_elite:]
            elites = [population[i] for i in elite_indices]
            new_population = elites.copy()
            
            while len(new_population) < self.pop_size:
                parent1, parent2 = self.select_func(population, fitnesses)  # 选择父母
                child = self.crossover_func(parent1, parent2)               # 交叉
                child = self.mutation_func(child)                           # 变异
                new_population.append(child)
            
            population = new_population
        
        final_fitnesses = [self.fitness_func(individual) for individual in population]
        best_individual = population[np.argmax(final_fitnesses)]
        
        return best_individual, best_fitness_history