import numpy as np
import random
from schedulers.base_scheduler import BaseScheduler
from schedulers.metaheuristic.fitness import FitnessCalculator
from utils.visualizer import plot_ga_convergence

class HGASAScheduler(BaseScheduler):

    def __init__(self, population_size, num_generations, mutation_rate, crossover_rate,
                 tournament_size=3, sa_initial_temp=100.0, sa_cooling_rate=0.95, sa_iterations=20):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        
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
            raise Exception("Fitness weights must be set before running HGA-SA scheduler")

        num_tasks = len(tasks)
        population = self._initialize_population(num_tasks)
        best_schedule = None
        best_fitness = float('inf')

        history = {'generation': [], 'gbest': [], 'temperature': []}
        temp = self.sa_initial_temp

        for generation in range(self.num_generations):
            fitnesses = [self.fitness_calculator.calculate_fitness(chrom) for chrom in population]
            
            gen_best_idx = np.argmin(fitnesses)
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_schedule = population[gen_best_idx][:]

            new_population = []
            for chrom in population:
                polished = self._run_sa(chrom, temp)
                new_population.append(polished)

            next_gen = [best_schedule]
            while len(next_gen) < self.population_size:
                p1 = self._selection(new_population, fitnesses)
                p2 = self._selection(new_population, fitnesses)

                if random.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                next_gen.append(self._mutation(c1))
                if len(next_gen) < self.population_size:
                    next_gen.append(self._mutation(c2))

            population = next_gen
            temp *= self.sa_cooling_rate

            history['generation'].append(generation + 1)
            history['gbest'].append(best_fitness)
            history['temperature'].append(temp)

        plot_ga_convergence(history, title="HGA-SA Convergence", filename="results/HGA_SA_convergence.png")
        return best_schedule

    def _initialize_population(self, num_tasks):
        return [[random.randint(0, self.num_vms - 1) for _ in range(num_tasks)] for _ in range(self.population_size)]

    def _selection(self, population, fitnesses):
        tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
        return min(tournament, key=lambda x: x[1])[0]

    def _crossover(self, p1, p2):
        point = random.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutation(self, chrom):
        for i in range(len(chrom)):
            if random.random() < self.mutation_rate:
                chrom[i] = random.randint(0, self.num_vms - 1)
        return chrom

    def _run_sa(self, initial_schedule, temp):
        current = initial_schedule[:]
        current_fitness = self.fitness_calculator.calculate_fitness(current)
        best = current[:]
        best_fitness = current_fitness
        current_temp = temp

        for _ in range(self.sa_iterations):
            neighbor = self._create_neighbor(current)
            neighbor_fitness = self.fitness_calculator.calculate_fitness(neighbor)

            if neighbor_fitness < best_fitness:
                best, best_fitness = neighbor[:], neighbor_fitness

            delta = neighbor_fitness - current_fitness
            if delta < 0 or random.random() < np.exp(-delta / (current_temp + 1e-9)):
                current, current_fitness = neighbor[:], neighbor_fitness

            current_temp *= self.sa_cooling_rate

        return best

    def _create_neighbor(self, schedule):
        neighbor = schedule[:]
        task_to_mutate = random.randint(0, len(schedule) - 1)
        neighbor[task_to_mutate] = random.randint(0, self.num_vms - 1)
        return neighbor