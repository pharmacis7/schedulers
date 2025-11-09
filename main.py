import os
import numpy as np
from tqdm import tqdm

from simulator.environment import Task, VM
from simulator.metrics import calculate_metrics
from utils.datagen import create_dag, create_vms
from utils.visualizer import plot_results, save_results_to_csv, plot_average_results

# import schedulers programs
from schedulers.simple.fcfs import FCFSScheduler
from schedulers.simple.rr import RRScheduler
from schedulers.metaheuristic.ga import GAScheduler
from schedulers.metaheuristic.pso import PSOScheduler



# import our algos
from schedulers.metaheuristic.test1 import test1
from schedulers.metaheuristic.test2 import test2

# 10 random test cases nodes -> 10,20,30...100
SCENARIOS = {} # empty dict
NUM_VMS = 10  # 10 vm const for every test case
for i in range(1, 11):
    num_tasks = i * 10
    scenario_name = f"{num_tasks}_Tasks_DAG" # unique name of TC as key -> for dict
    SCENARIOS[scenario_name] = (num_tasks, NUM_VMS) 

# define schedulers to test vm
SCHEDULERS = {
    "FCFS": FCFSScheduler(),
    "Round Robin": RRScheduler(),
    "GA": GAScheduler(
        population_size=50,
        num_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    ),
    "PSO": PSOScheduler(
        swarm_size=50,
        num_iterations=100,
        w=0.9,
        c1=1.5,
        c2=1.5
    ),
    # --- MODIFIED: Include both Test1 and Test2 ---
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
        num_elite_sa=5,         # Polish the top 5 solutions
        sa_initial_temp=50.0,   # Lower temp
        sa_cooling_rate=0.9,
        sa_iterations=30        # Quick polish
    )
}

# multiobjective fitness function weights -> defint here
FITNESS_WEIGHTS = {
    'w_makespan': 0.5,
    'w_cost': 0.25,
    'w_energy': 0.25
}

# simulation loop -> iterate over all TC and all schedulers to get relevant measures
def run_simulation():
    print("starting DAG simulation .................")
    
    # create directories -> if they dont exist if exist then it probably not generate errors
    os.makedirs("results", exist_ok=True)
    os.makedirs("test_cases", exist_ok=True)
    os.makedirs("average_result", exist_ok=True)
    
    # Dictionary to collect all results for averaging
    all_results_collection = {scheduler_name: [] for scheduler_name in SCHEDULERS.keys()}

    for scenario_name, (num_tasks, num_vms) in SCENARIOS.items():
        print(f"\n--- Running Scenario: {scenario_name} ({num_tasks} Tasks, {num_vms} VMs) ---")

        # random DAG generate, random specs VM generate
        np.random.seed(42)
        tasks = create_dag(num_tasks, scenario_name) 
        vms = create_vms(num_vms)
        
        scenario_results = {}

        # run schedulers instances imported above so each random DAG , VM spec set run for all schedulers
        for scheduler_name, scheduler_instance in SCHEDULERS.items():
            print(f"  running {scheduler_name}...")
            
            # first checking if scheduler support set_fitness_weights chain function if yes then pass if no then dont
            if hasattr(scheduler_instance, 'set_fitness_weights'):
                scheduler_instance.set_fitness_weights(FITNESS_WEIGHTS, tasks, vms)
                
            # tqdm -> for progress bar nothgin else 
            with tqdm(total=1, desc=f"  {scheduler_name}", leave=False) as pbar:
                schedule = scheduler_instance.schedule(tasks, vms)
                pbar.update(1)

            # call claculate_metrics method here saved results in metrics var
            metrics = calculate_metrics(schedule, tasks, vms) 

            # update scenario_results
            scenario_results[scheduler_name] = metrics
            
            # also push in all_results_collection dict
            all_results_collection[scheduler_name].append(metrics)

        # 4. Save and Plot Results for this scenario
        plot_filename = f"results/{scenario_name}_results.png"
        plot_results(scenario_results, title=f"{scenario_name} ({num_tasks} Tasks, {num_vms} VMs)", filename=plot_filename)
        # print(f"\n  Results plot saved to {plot_filename}")
        
        csv_filename = f"results/{scenario_name}_results.csv"
        save_results_to_csv(scenario_results, filename=csv_filename)
        # print(f"  Results data saved to {csv_filename}")

    # after test cases loop ends -> now generate average results across all scenarios
    
    avg_plot_file = "average_result/average_comparison.png"
    plot_average_results(all_results_collection, filename=avg_plot_file)


# This is the "trigger" that calls the function
if __name__ == "__main__":
    run_simulation()