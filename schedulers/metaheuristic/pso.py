import numpy as np
import random
from schedulers.base_scheduler import BaseScheduler
from schedulers.metaheuristic.fitness import FitnessCalculator

class Particle:
    def __init__(self, num_tasks, num_vms):
        self.num_vms = num_vms
        self.position = np.random.uniform(0, num_vms - 1, num_tasks)
        self.velocity = np.random.uniform(-1, 1, num_tasks)
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = float('inf')

    def get_discrete_schedule(self):
        clamped_pos = np.clip(self.position, 0, self.num_vms - 1)
        return np.round(clamped_pos).astype(int)

class PSOScheduler(BaseScheduler):
    
    def __init__(self, swarm_size, num_iterations, w, c1, c2):
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.fitness_calculator = None

    def set_fitness_weights(self, weights, tasks, vms):
        self.fitness_calculator = FitnessCalculator(tasks, vms, weights)

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set for PSO scheduler")
        
        num_tasks = len(tasks)
        num_vms = len(vms)
        
        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        
        gbest_position = None
        gbest_fitness = float('inf')

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