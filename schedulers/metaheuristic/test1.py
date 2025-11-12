import numpy as np
import random
from schedulers.metaheuristic.pso import PSOScheduler, Particle
from schedulers.metaheuristic.fitness import FitnessCalculator

class test1(PSOScheduler):
    
    def __init__(self, swarm_size, num_iterations, w, c1, c2,
                 stagnation_threshold, sa_initial_temp, sa_cooling_rate, sa_iterations):
        
        super().__init__(swarm_size, num_iterations, w, c1, c2)
        
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
        
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set for MO-PSO-SA scheduler")
        
        num_tasks = len(tasks)
        num_vms = len(vms)
        
        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        
        gbest_position = None
        gbest_fitness = float('inf')
        
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')

        for _ in range(self.num_iterations):
            for particle in swarm:
                schedule = particle.get_discrete_schedule()
                fitness = self.fitness_calculator.calculate_fitness(schedule)
                
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                    
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = np.copy(particle.position)
            
            if gbest_position is None:
                gbest_position = swarm[0].pbest_position

            if gbest_fitness < self.last_best_fitness:
                self.stagnation_counter = 0
                self.last_best_fitness = gbest_fitness
            else:
                self.stagnation_counter += 1

            if self.stagnation_counter >= self.stagnation_threshold:
                current_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int).tolist()
                
                new_schedule = self._run_sa(current_schedule, num_vms)
                new_fitness = self.fitness_calculator.calculate_fitness(new_schedule)
                
                if new_fitness < gbest_fitness:
                    gbest_fitness = new_fitness
                    gbest_position = np.array(new_schedule)
                    
                self.stagnation_counter = 0

            for particle in swarm:
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                
                cognitive_velocity = self.c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                
                particle.position = particle.position + particle.velocity
                particle.position = np.clip(particle.position, 0, num_vms - 1)
        
        final_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int)
        return final_schedule.tolist()

    def _run_sa(self, initial_schedule, num_vms):
        current_temp = self.sa_initial_temp
        current_schedule = initial_schedule[:]
        current_fitness = self.fitness_calculator.calculate_fitness(current_schedule)
        
        best_schedule = current_schedule[:]
        best_fitness = current_fitness
        
        for _ in range(self.sa_iterations):
            neighbor_schedule = self._create_neighbor(current_schedule, num_vms)
            neighbor_fitness = self.fitness_calculator.calculate_fitness(neighbor_schedule)
            
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_schedule = neighbor_schedule[:]
            
            delta_fitness = neighbor_fitness - current_fitness
            
            if delta_fitness < 0:
                current_schedule = neighbor_schedule[:]
                current_fitness = neighbor_fitness
            else:
                acceptance_prob = np.exp(-delta_fitness / current_temp)
                if random.random() < acceptance_prob:
                    current_schedule = neighbor_schedule[:]
                    current_fitness = neighbor_fitness
                    
            current_temp *= self.sa_cooling_rate
            
        return best_schedule

    def _create_neighbor(self, schedule, num_vms):
        neighbor = schedule[:]
        task_to_mutate = random.randint(0, len(schedule) - 1)
        new_vm = random.randint(0, num_vms - 1)
        neighbor[task_to_mutate] = new_vm
        return neighbor