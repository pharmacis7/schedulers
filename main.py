import os
import numpy as np
from tqdm import tqdm

from simulator.environment import Task, VM
from simulator.metrics import calculate_metrics
from utils.datagen import create_dag, create_vms
from utils.visualizer import plot_results, save_results_to_csv, plot_average_results

# Import schedulers
from schedulers.simple.fcfs import FCFSScheduler
from schedulers.simple.rr import RRScheduler
from schedulers.metaheuristic.ga import GAScheduler
from schedulers.metaheuristic.pso import PSOScheduler
from schedulers.metaheuristic.paper import BasePaperPSO      # ⬅️ Renamed
from schedulers.metaheuristic.paperGA import BasePaperGA     # ⬅️ Renamed
from schedulers.metaheuristic.test1 import test1             # Improved PSO + SA
from schedulers.metaheuristic.test2 import test2             # Improved GA + SA

# -----------------------------------------------------------
# Simulation setup
# -----------------------------------------------------------

SCENARIOS = {}
NUM_VMS = 10
for i in range(1, 11):
    num_tasks = i * 10
    SCENARIOS[f"{num_tasks}_Tasks_DAG"] = (num_tasks, NUM_VMS)

# Order of Schedulers — GA → BasePaper(GA) → PSO → BasePaper(PSO) → test1 → test2
SCHEDULERS = {
    "FCFS": FCFSScheduler(),
    "Round Robin": RRScheduler(),

    # --- GA variants ---
    "GA": GAScheduler(
        population_size=50,
        num_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    ),
    "BasePaper(GA)": BasePaperGA(
        population_size=50,
        num_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        sa_initial_temp=100.0,
        sa_cooling_rate=0.95,
        sa_iterations=20
    ),

    # --- PSO variants ---
    "PSO": PSOScheduler(
        swarm_size=50,
        num_iterations=100,
        w=0.9,
        c1=1.5,
        c2=1.5
    ),
    "BasePaper(PSO)": BasePaperPSO(
        swarm_size=50,
        num_iterations=100,
        w=0.9,
        c1=1.5,
        c2=1.5,
        sa_initial_temp=100.0,
        sa_cooling_rate=0.95
    ),

    # --- Improved Hybrids ---
    "test1": test1(
        swarm_size=50,
        num_iterations=100,
        w=0.9,
        c1=1.5,
        c2=1.5,
        stagnation_threshold=15,
        sa_initial_temp=100.0,
        sa_cooling_rate=0.95,
        sa_iterations=50
    ),
    "test2": test2(
        population_size=50,
        num_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        num_elite_sa=5,
        sa_initial_temp=50.0,
        sa_cooling_rate=0.9,
        sa_iterations=30
    )
}

# Fitness Weights
FITNESS_WEIGHTS = {
    'w_makespan': 0.5,
    'w_cost': 0.25,
    'w_energy': 0.25
}


def run_simulation():
    print("🚀 Starting DAG simulation...")

    os.makedirs("results", exist_ok=True)
    os.makedirs("test_cases", exist_ok=True)
    os.makedirs("average_result", exist_ok=True)

    all_results_collection = {scheduler_name: [] for scheduler_name in SCHEDULERS.keys()}

    for scenario_name, (num_tasks, num_vms) in SCENARIOS.items():
        print(f"\n--- Running Scenario: {scenario_name} ({num_tasks} Tasks, {num_vms} VMs) ---")

        np.random.seed(42)
        tasks = create_dag(num_tasks, scenario_name)
        vms = create_vms(num_vms)

        scenario_results = {}

        for scheduler_name, scheduler_instance in SCHEDULERS.items():
            print(f"  ▶ Running {scheduler_name}...")

            if hasattr(scheduler_instance, 'set_fitness_weights'):
                scheduler_instance.set_fitness_weights(FITNESS_WEIGHTS, tasks, vms)
            
            with tqdm(total=1, desc=f"  {scheduler_name}", leave=False) as pbar:
                schedule = scheduler_instance.schedule(tasks, vms)
                pbar.update(1)

            metrics = calculate_metrics(schedule, tasks, vms)
            scenario_results[scheduler_name] = metrics
            all_results_collection[scheduler_name].append(metrics)

        # Save per-scenario results
        plot_filename = f"results/{scenario_name}_results.png"
        plot_results(
            scenario_results,
            title=f"{scenario_name} ({num_tasks} Tasks, {num_vms} VMs)",
            filename=plot_filename
        )

        csv_filename = f"results/{scenario_name}_results.csv"
        save_results_to_csv(scenario_results, filename=csv_filename)

    # Average across all test cases
    avg_plot_file = "average_result/average_comparison.png"
    plot_average_results(all_results_collection, filename=avg_plot_file)
    print(f"\n✅ Simulation complete! Average comparison saved at {avg_plot_file}")


if __name__ == "__main__":
    run_simulation()
