import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from simulator.environment import Task, VM

def create_dag(num_tasks, scenario_name, density=0.2):
    
    # generate DAG structure
    G = nx.DiGraph() # empty directed graph
    G.add_nodes_from(range(num_tasks)) # create num_tasks vertexes in graph

    # for each possible pair of i,j -> create graph edges i to j such that i < j only {ensure no cycle}
    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            # generate a rand num [0,1) if < density -> create edge {thats how create random edges}
            if np.random.rand() < density:
                G.add_edge(i, j)

    # create task objects
    tasks = []
    task_lengths = np.random.randint(1000, 10000, num_tasks) # random task MIPS 1000 to 9999 
    task_ram = np.random.randint(256, 2049, num_tasks) # random task RAM 256mb to 2048mb
    
    # fill in tasks data structure 
    for i in range(num_tasks):
        tasks.append(Task(
            task_id=i, 
            length=task_lengths[i], 
            ram_required=task_ram[i]
        ))
        
    # fill in parents and children array in tasks
    for node in G.nodes():
        tasks[node].parents = set(G.predecessors(node))
        tasks[node].children = set(G.successors(node))
        
        # For each parent, assign a data size (e.g., 50MB to 500MB)
        # it means a children task cant start untill it recv this data from parent
        for p_id in tasks[node].parents:
            data_size = np.random.randint(50, 501)
            tasks[node].parent_data_sizes[p_id] = data_size
            
    # Save visualization
    plt.figure(figsize=(12, 8))
    try:
        # Try to use the 'dot' layout if pydot/graphviz is installed
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        # Fallback to a simpler layout
        print("  > pydot/graphviz not found. Using kamada_kawai_layout for DAG image.")
        pos = nx.kamada_kawai_layout(G)
        
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, arrows=True)
    plt.title(f"Test Case: {scenario_name} ({num_tasks} Tasks DAG)")
    
    
    # save task image
    image_filename = f"test_cases/{scenario_name}.png"
    plt.savefig(image_filename)
    plt.close()
        
    return tasks

# generate 10 vms with different specs
def create_vms(num_vms):
    vms = []
    np.random.seed(42) # lock seed var so np generate -> so that when main execute it generate same set of 10 random vms for each test case
    
    # vm mips -> speed array
    mips = np.random.randint(1000, 4000, num_vms)
    
    # vm cost per second array
    costs = np.random.uniform(0.01, 0.15, num_vms)
    
    # vm power consumption array
    power = np.random.uniform(100, 300, num_vms)
    
    # vm ram cap array
    ram_capacities = np.random.choice([2048, 4096, 8192], num_vms)
    
    # vm bandwidth cap array
    bandwidths = np.random.choice([100, 500, 1000], num_vms)
    
    # fill in vms data structure
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