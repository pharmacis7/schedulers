import numpy as np
import random
from schedulers.base_scheduler import BaseScheduler
from schedulers.metaheuristic.fitness import FitnessCalculator

class test2(BaseScheduler):
    
    def __init__(self, population_size, num_generations, mutation_rate, crossover_rate, 
                 tournament_size=3, num_elite_sa=5, sa_initial_temp=100.0, 
                 sa_cooling_rate=0.95, sa_iterations=20):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
        self.num_elite_sa = num_elite_sa
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations
        
        self.fitness_calculator = None
        self.num_vms = 0

    def set_fitness_weights(self, weights, tasks, vms):
        self.fitness_calculator = FitnessCalculator(tasks, vms, weights)
        self.num_vms = len(vms)

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set for GA-SA scheduler")

        num_tasks = len(tasks)
        
        population = self._initialize_population(num_tasks)
        
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
                
                new_population.append(self._mutation(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutation(child2))
            
            new_fitnesses = [(self.fitness_calculator.calculate_fitness(p), i) for i, p in enumerate(new_population)]
            new_fitnesses.sort(key=lambda x: x[0])
            
            for i in range(self.num_elite_sa):
                if i >= len(new_population):
                    break
                
                elite_index = new_fitnesses[i][1]
                elite_solution = new_population[elite_index]
                
                polished_solution = self._run_sa(elite_solution)
                
                new_population[elite_index] = polished_solution

            population = new_population

        return best_schedule

    def _initialize_population(self, num_tasks):
        population = []
        for _ in range(self.population_size):
            chromosome = [random.randint(0, self.num_vms - 1) for _ in range(num_tasks)]
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

    def _mutation(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.num_vms - 1)
        return chromosome

    def _run_sa(self, initial_schedule):
        current_temp = self.sa_initial_temp
        current_schedule = initial_schedule[:]
        current_fitness = self.fitness_calculator.calculate_fitness(current_schedule)
        
        best_schedule = current_schedule[:]
        best_fitness = current_fitness
        
        for _ in range(self.sa_iterations):
            neighbor_schedule = self._create_neighbor(current_schedule)
            neighbor_fitness = self.fitness_calculator.calculate_fitness(neighbor_schedule)
            
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_schedule = neighbor_schedule[:]
            
            delta_fitness = neighbor_fitness - current_fitness
            
            if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / current_temp):
                current_schedule = neighbor_schedule[:]
                current_fitness = neighbor_fitness
                    
            current_temp *= self.sa_cooling_rate
            
        return best_schedule

    def _create_neighbor(self, schedule):
        neighbor = schedule[:]
        task_to_mutate = random.randint(0, len(schedule) - 1)
        new_vm = random.randint(0, self.num_vms - 1)
        neighbor[task_to_mutate] = new_vm
        return neighbor