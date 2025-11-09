import numpy as np
import random
from schedulers.base_scheduler import BaseScheduler
from schedulers.metaheuristic.fitness import FitnessCalculator

class Particle:
    def __init__(self, num_tasks, num_vms):
        self.num_vms = num_vms
        
        # Continuous position vector
        self.position = np.random.uniform(0, num_vms - 1, num_tasks)
        
        # Velocity vector
        self.velocity = np.random.uniform(-1, 1, num_tasks)
        
        # Personal best
        self.pbest_position = np.copy(self.position)
        self.pbest_fitness = float('inf')

    def get_discrete_schedule(self):
        """Converts continuous position to a discrete schedule."""
        # Clamp positions to be within valid VM indices
        clamped_pos = np.clip(self.position, 0, self.num_vms - 1)
        # Round to nearest integer to get VM IDs
        return np.round(clamped_pos).astype(int)

class PSOScheduler(BaseScheduler):
    """
    Particle Swarm Optimization (PSO) Scheduler.
    
    Uses a continuous-position PSO, which is discretized
    to evaluate fitness.
    """
    
    def __init__(self, swarm_size, num_iterations, w, c1, c2):
        self.swarm_size = swarm_size
        self.num_iterations = num_iterations
        self.w = w    # Inertia weight
        self.c1 = c1  # Cognitive (personal) coefficient
        self.c2 = c2  # Social (global) coefficient
        self.fitness_calculator = None

    def set_fitness_weights(self, weights, tasks, vms):
        self.fitness_calculator = FitnessCalculator(tasks, vms, weights)

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set for PSO scheduler")
        
        num_tasks = len(tasks)
        num_vms = len(vms)
        
        # 1. Initialize Swarm
        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        
        gbest_position = None
        gbest_fitness = float('inf')

        for _ in range(self.num_iterations):
            for particle in swarm:
                # 2. Evaluate Fitness
                schedule = particle.get_discrete_schedule()
                fitness = self.fitness_calculator.calculate_fitness(schedule)
                
                # 3. Update pBest
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                    
                # 4. Update gBest
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = np.copy(particle.position)
            
            if gbest_position is None:
                gbest_position = swarm[0].pbest_position # Initialize gbest
            
            # 5. Update Particle Velocities and Positions
            for particle in swarm:
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive_velocity = self.c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                
                # x = x + v
                particle.position = particle.position + particle.velocity
                
                # Clamp position to valid range (0 to num_vms-1)
                particle.position = np.clip(particle.position, 0, num_vms - 1)
        
        # Return the best schedule found
        final_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int)
        return final_schedule.tolist()