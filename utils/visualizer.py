import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_results(results, title, filename):
    
    schedulers = list(results.keys())
    metrics = ['makespan', 'total_cost', 'total_energy']
    metric_names = ['Makespan (s)', 'Total Cost ($)', 'Total Energy (Wh)']
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
    fig.suptitle(title, fontsize=16)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        values = [results[s][metric] for s in schedulers]
        
        x_pos = np.arange(len(schedulers))
        bars = ax.bar(x_pos, values, align='center',
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(schedulers, rotation=45, ha='right')
        ax.set_ylabel(name)
        ax.set_title(f'Comparison of {name}')
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval,
                    f'{yval:.2f}', va='bottom', ha='center') 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)

def save_results_to_csv(results, filename):
    
    header = ['Scheduler', 'Makespan', 'TotalCost', 'TotalEnergy']
    
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
            print(f"Warning: No results found for {scheduler_name}. Skipping.")
            continue
            
        # Calculate the mean for each metric
        avg_makespan = np.mean([m['makespan'] for m in metric_list])
        avg_cost = np.mean([m['total_cost'] for m in metric_list])
        avg_energy = np.mean([m['total_energy'] for m in metric_list])
        
        average_metrics[scheduler_name] = {
            'makespan': avg_makespan,
            'total_cost': avg_cost,
            'total_energy': avg_energy
        }
        
    # --- Now we reuse our existing functions ---
    
    # 1. Plot the averages
    plot_results(
        average_metrics,
        title="Average Performance Across All Test Cases",
        filename=filename
    )
    
    # 2. Save the average data to a CSV
    csv_filename = filename.replace(".png", ".csv")
    save_results_to_csv(average_metrics, csv_filename)