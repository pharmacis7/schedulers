import numpy as np
import random
from schedulers.metaheuristic.pso import PSOScheduler, Particle
from schedulers.metaheuristic.fitness import FitnessCalculator

class test1(PSOScheduler):
    """
    Novel Algorithm: Multi-Objective PSO with Simulated Annealing (MO-PSO-SA).
    
    This algorithm runs PSO, but when it detects stagnation (the
    global best solution hasn't improved), it triggers a
    Simulated Annealing local search on the global best solution
    to "jiggle" it out of the local optimum.
    """
    
    def __init__(self, swarm_size, num_iterations, w, c1, c2,
                 stagnation_threshold, sa_initial_temp, sa_cooling_rate, sa_iterations):
        
        super().__init__(swarm_size, num_iterations, w, c1, c2)
        
        # Stagnation parameters
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')
        
        # Simulated Annealing (SA) parameters
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_iterations = sa_iterations

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set for MO-PSO-SA scheduler")
        
        num_tasks = len(tasks)
        num_vms = len(vms)
        
        # 1. Initialize Swarm
        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        
        gbest_position = None
        gbest_fitness = float('inf')
        
        self.stagnation_counter = 0
        self.last_best_fitness = float('inf')

        for _ in range(self.num_iterations):
            for particle in swarm:
                # 2. Evaluate Fitness
                schedule = particle.get_discrete_schedule()
                fitness = self.fitness_calculator.calculate_fitness(schedule)
                
                # 3. Update pBest and gBest
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                    
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = np.copy(particle.position)
            
            if gbest_position is None:
                gbest_position = swarm[0].pbest_position

            # 5. Check for Stagnation
            if gbest_fitness < self.last_best_fitness:
                self.stagnation_counter = 0  # Improvement, reset counter
                self.last_best_fitness = gbest_fitness
            else:
                self.stagnation_counter += 1

            # 6. Activate SA if stagnated
            if self.stagnation_counter >= self.stagnation_threshold:
                # print(f"  (Stagnation detected! Running SA...)")
                
                # Run SA on the *discrete schedule* of the gbest
                current_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int).tolist()
                
                new_schedule = self._run_sa(current_schedule, num_vms)
                new_fitness = self.fitness_calculator.calculate_fitness(new_schedule)
                
                # If SA found an even better solution, update gbest
                if new_fitness < gbest_fitness:
                    gbest_fitness = new_fitness
                    # We must update gbest_position (continuous) as well
                    # We'll set it to the new schedule (it will be discretized again anyway)
                    gbest_position = np.array(new_schedule)
                    
                # Reset counter regardless
                self.stagnation_counter = 0

            # 7. Update Particle Velocities and Positions (Standard PSO step)
            for particle in swarm:
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                
                cognitive_velocity = self.c1 * r1 * (particle.pbest_position - particle.position)
                social_velocity = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_velocity + social_velocity
                
                particle.position = particle.position + particle.velocity
                particle.position = np.clip(particle.position, 0, num_vms - 1)
        
        # Return the best discrete schedule found
        final_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int)
        return final_schedule.tolist()

    def _run_sa(self, initial_schedule, num_vms):
        """
        Performs the Simulated Annealing local search.
        """
        current_temp = self.sa_initial_temp
        current_schedule = initial_schedule[:]
        current_fitness = self.fitness_calculator.calculate_fitness(current_schedule)
        
        best_schedule = current_schedule[:]
        best_fitness = current_fitness
        
        for _ in range(self.sa_iterations):
            # Create a neighbor solution (small change)
            neighbor_schedule = self._create_neighbor(current_schedule, num_vms)
            neighbor_fitness = self.fitness_calculator.calculate_fitness(neighbor_schedule)
            
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_schedule = neighbor_schedule[:]
            
            # Metropolis acceptance criterion
            delta_fitness = neighbor_fitness - current_fitness
            
            if delta_fitness < 0:
                # Better solution, always accept
                current_schedule = neighbor_schedule[:]
                current_fitness = neighbor_fitness
            else:
                # Worse solution, accept with probability
                acceptance_prob = np.exp(-delta_fitness / current_temp)
                if random.random() < acceptance_prob:
                    current_schedule = neighbor_schedule[:]
                    current_fitness = neighbor_fitness
                    
            # Cool down
            current_temp *= self.sa_cooling_rate
            
        return best_schedule

    def _create_neighbor(self, schedule, num_vms):
        """Creates a neighbor schedule by changing one task's VM."""
        neighbor = schedule[:]
        task_to_mutate = random.randint(0, len(schedule) - 1)
        new_vm = random.randint(0, num_vms - 1)
        neighbor[task_to_mutate] = new_vm
        return neighbor