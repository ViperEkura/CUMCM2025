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