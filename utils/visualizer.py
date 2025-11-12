import matplotlib.pyplot as plt
import numpy as np
import csv
import os

# -----------------------------------------------------------
# General Metrics Visualization
# -----------------------------------------------------------

def plot_results(results, title, filename):
    # Enforce display order in bar plots
    scheduler_order = [
        "FCFS", "Round Robin",
        "GA", "BasePaper(GA)",
        "PSO", "BasePaper(PSO)",
        "test1", "test2"
    ]
    
    schedulers = [s for s in scheduler_order if s in results]
    metrics = ['makespan', 'total_cost', 'total_energy']
    metric_names = ['Makespan (s)', 'Total Cost ($)', 'Total Energy (Wh)']

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        values = [results[s][metric] for s in schedulers]
        x_pos = np.arange(len(schedulers))

        bars = ax.bar(x_pos, values, align='center',
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
                             '#17becf', '#e377c2', '#8c564b', '#7f7f7f'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(schedulers, rotation=45, ha='right')
        ax.set_ylabel(name)
        ax.set_title(f'Comparison of {name}')

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval,
                    f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def save_results_to_csv(results, filename):
    header = ['Scheduler', 'Makespan', 'TotalCost', 'TotalEnergy']
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for scheduler_name, metrics in results.items():
            writer.writerow([
                scheduler_name,
                f"{metrics['makespan']:.4f}",
                f"{metrics['total_cost']:.4f}",
                f"{metrics['total_energy']:.4f}"
            ])


def plot_average_results(all_results, filename):
    average_metrics = {}
    for scheduler_name, metric_list in all_results.items():
        if not metric_list:
            continue
        avg_makespan = np.mean([m['makespan'] for m in metric_list])
        avg_cost = np.mean([m['total_cost'] for m in metric_list])
        avg_energy = np.mean([m['total_energy'] for m in metric_list])
        average_metrics[scheduler_name] = {
            'makespan': avg_makespan,
            'total_cost': avg_cost,
            'total_energy': avg_energy
        }

    plot_results(average_metrics,
                 title="Average Performance Across All Test Cases",
                 filename=filename)
    csv_filename = filename.replace(".png", ".csv")
    save_results_to_csv(average_metrics, csv_filename)

# -----------------------------------------------------------
# Convergence Plots for Paper-style Hybrids
# -----------------------------------------------------------

def plot_paper_convergence(history, title="BasePaper(PSO) Convergence", filename="results/paper_pso_convergence.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    iters = history.get('iteration', [])
    gbest = history.get('gbest', [])
    temps = history.get('temperature', [])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Global Best Fitness", color='tab:blue', fontsize=12)
    ax1.plot(iters, gbest, label="Global Best Fitness", color='tab:blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if temps:
        ax2 = ax1.twinx()
        ax2.plot(iters, temps, label="Temperature", color='tab:red', linestyle='--', linewidth=2)
        ax2.set_ylabel("Temperature (SA)", color='tab:red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_ga_convergence(history, title="BasePaper(GA) Convergence", filename="results/paper_ga_convergence.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    gens = history.get('generation', [])
    gbest = history.get('gbest', [])
    temps = history.get('temperature', [])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Global Best Fitness", color='tab:blue', fontsize=12)
    ax1.plot(gens, gbest, color='tab:blue', linewidth=2, label='Global Best Fitness')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if temps:
        ax2 = ax1.twinx()
        ax2.plot(gens, temps, color='tab:red', linestyle='--', linewidth=2, label='Temperature')
        ax2.set_ylabel("Temperature (SA)", color='tab:red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
