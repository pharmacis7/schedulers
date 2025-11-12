import numpy as np
import random
from schedulers.metaheuristic.pso import PSOScheduler, Particle
from schedulers.metaheuristic.fitness import FitnessCalculator
from utils.visualizer import plot_paper_convergence

class BasePaperPSO(PSOScheduler):
    """
    BasePaper(PSO): Hybrid PSO + SA (Paper-style, same structure as PSO).
    
    - Standard PSO loop maintained.
    - Adds SA-based acceptance rule for slightly worse pBest updates.
    - Temperature cools each iteration.
    """

    def __init__(self, swarm_size, num_iterations, w, c1, c2,
                 sa_initial_temp, sa_cooling_rate):
        super().__init__(swarm_size, num_iterations, w, c1, c2)
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set before running BasePaper(PSO) scheduler")

        num_tasks = len(tasks)
        num_vms = len(vms)

        # 1️⃣ Initialize swarm (same as PSO)
        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        gbest_position = None
        gbest_fitness = float('inf')

        # Initialize SA temperature
        temp = self.sa_initial_temp

        # Track convergence
        history = {'iteration': [], 'gbest': [], 'temperature': []}

        # 2️⃣ Main PSO loop (identical structure)
        for iteration in range(self.num_iterations):
            for particle in swarm:
                # Evaluate fitness
                schedule = particle.get_discrete_schedule()
                fitness = self.fitness_calculator.calculate_fitness(schedule)

                # SA-based personal best update
                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                else:
                    # SA acceptance: may accept worse pBest
                    delta = fitness - particle.pbest_fitness
                    acceptance_prob = np.exp(-delta / (temp + 1e-9))
                    if random.random() < acceptance_prob:
                        particle.pbest_fitness = fitness
                        particle.pbest_position = np.copy(particle.position)

                # Update global best
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest_position = np.copy(particle.position)

            if gbest_position is None:
                gbest_position = swarm[0].pbest_position

            # Velocity and position updates (standard PSO)
            for particle in swarm:
                r1 = np.random.rand(num_tasks)
                r2 = np.random.rand(num_tasks)
                cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
                social = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, 0, num_vms - 1)

            # Cool down temperature
            temp *= self.sa_cooling_rate

            # Record convergence
            history['iteration'].append(iteration + 1)
            history['gbest'].append(gbest_fitness)
            history['temperature'].append(temp)

        # Plot convergence
        plot_paper_convergence(history, filename="results/BasePaperPSO_convergence.png")

        final_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int)
        return final_schedule.tolist()
