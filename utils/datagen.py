import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from simulator.environment import Task, VM

def create_dag(num_tasks, scenario_name, density=0.2):
    
    G = nx.DiGraph()
    G.add_nodes_from(range(num_tasks))

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if np.random.rand() < density:
                G.add_edge(i, j)

    tasks = []
    task_lengths = np.random.randint(1000, 10000, num_tasks) 
    task_ram = np.random.randint(256, 2049, num_tasks)
    
    for i in range(num_tasks):
        tasks.append(Task(
            task_id=i, 
            length=task_lengths[i], 
            ram_required=task_ram[i]
        ))
        
    for node in G.nodes():
        tasks[node].parents = set(G.predecessors(node))
        tasks[node].children = set(G.successors(node))
        
        for p_id in tasks[node].parents:
            data_size = np.random.randint(50, 501)
            tasks[node].parent_data_sizes[p_id] = data_size
            
    plt.figure(figsize=(12, 8))
    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        print("  > pydot/graphviz not found. Using kamada_kawai_layout for DAG image.")
        pos = nx.kamada_kawai_layout(G)
        
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, arrows=True)
    plt.title(f"Test Case: {scenario_name} ({num_tasks} Tasks DAG)")
    
    image_filename = f"test_cases/{scenario_name}.png"
    plt.savefig(image_filename)
    plt.close()
        
    return tasks

def create_vms(num_vms):
    vms = []
    np.random.seed(42)
    
    mips = np.random.randint(1000, 4000, num_vms)
    costs = np.random.uniform(0.01, 0.15, num_vms)
    power = np.random.uniform(100, 300, num_vms)
    ram_capacities = np.random.choice([2048, 4096, 8192], num_vms)
    bandwidths = np.random.choice([100, 500, 1000], num_vms)
    
    for i in range(num_vms):
        vms.append(VM(
            vm_id=i, 
            mips=mips[i], 
            cost_per_sec=costs[i], 
            power=power[i],
            ram_capacity=ram_capacities[i],
            bandwidth=bandwidths[i]
        ))
        
    return vms