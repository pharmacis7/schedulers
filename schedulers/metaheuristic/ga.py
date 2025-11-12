import numpy as np
import random
from schedulers.base_scheduler import BaseScheduler
from schedulers.metaheuristic.fitness import FitnessCalculator

class GAScheduler(BaseScheduler):
    
    def __init__(self, population_size, num_generations, mutation_rate, crossover_rate, tournament_size=3):
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.fitness_calculator = None

    def set_fitness_weights(self, weights, tasks, vms):
        self.fitness_calculator = FitnessCalculator(tasks, vms, weights)

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("fitness weights must be set for GA scheduler")

        num_tasks = len(tasks)
        num_vms = len(vms)
        
        population = self._initialize_population(num_tasks, num_vms)
        
        best_schedule = None
        best_fitness = float('inf')

        for _ in range(self.num_generations):
            fitnesses = [self.fitness_calculator.calculate_fitness(chrom) for chrom in population]
            
            current_best_idx = np.argmin(fitnesses)
            if fitnesses[current_best_idx] < best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_schedule = population[current_best_idx][:]

            new_population = []
            
            new_population.append(best_schedule)
            
            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitnesses)
                parent2 = self._selection(population, fitnesses)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                child1 = self._mutation(child1, num_vms)
                child2 = self._mutation(child2, num_vms)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population

        return best_schedule

    def _initialize_population(self, num_tasks, num_vms):
        population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, num_vms - 1) for _ in range(num_tasks)]
            population.append(chromosome)
        return population

    def _selection(self, population, fitnesses):
        tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
        winner = min(tournament, key=lambda x: x[1])[0]
        return winner

    def _crossover(self, parent1, parent2):
        num_tasks = len(parent1)
        point = random.randint(1, num_tasks - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutation(self, chromosome, num_vms):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, num_vms - 1)
        return chromosome