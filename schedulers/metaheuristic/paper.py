import numpy as np
import random
from schedulers.metaheuristic.pso import PSOScheduler, Particle
from schedulers.metaheuristic.fitness import FitnessCalculator
from utils.visualizer import plot_paper_convergence

class HPSOSAScheduler(PSOScheduler):

    def __init__(self, swarm_size, num_iterations, w, c1, c2,
                 sa_initial_temp, sa_cooling_rate):
        super().__init__(swarm_size, num_iterations, w, c1, c2)
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate

    def schedule(self, tasks, vms):
        if self.fitness_calculator is None:
            raise Exception("Fitness weights must be set before running HPSO-SA scheduler")

        num_tasks = len(tasks)
        num_vms = len(vms)

        swarm = [Particle(num_tasks, num_vms) for _ in range(self.swarm_size)]
        gbest_position = None
        gbest_fitness = float('inf')

        temp = self.sa_initial_temp

        history = {'iteration': [], 'gbest': [], 'temperature': []}

        for iteration in range(self.num_iterations):
            for particle in swarm:
                schedule = particle.get_discrete_schedule()
                fitness = self.fitness_calculator.calculate_fitness(schedule)

                if fitness < particle.pbest_fitness:
                    particle.pbest_fitness = fitness
                    particle.pbest_position = np.copy(particle.position)
                else:
                    delta = fitness - particle.pbest_fitness
                    acceptance_prob = np.exp(-delta / (temp + 1e-9))
                    if random.random() < acceptance_prob:
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
                cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
                social = self.c2 * r2 * (gbest_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position += particle.velocity
                particle.position = np.clip(particle.position, 0, num_vms - 1)

            temp *= self.sa_cooling_rate

            history['iteration'].append(iteration + 1)
            history['gbest'].append(gbest_fitness)
            history['temperature'].append(temp)

        plot_paper_convergence(history, title="HPSO-SA Convergence", filename="results/HPSO_SA_convergence.png")

        final_schedule = np.round(np.clip(gbest_position, 0, num_vms - 1)).astype(int)
        return final_schedule.tolist()