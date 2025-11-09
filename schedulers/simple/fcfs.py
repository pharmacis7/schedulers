import networkx as nx
from schedulers.base_scheduler import BaseScheduler

class FCFSScheduler(BaseScheduler):

    def schedule(self, tasks, vms):
        num_vms = len(vms)
        num_tasks = len(tasks)
        schedule = [0] * num_tasks
        
        # get a fresh graph structure from tasks data (cz it includes parents and children node sets)
        G = nx.DiGraph()
        G.add_nodes_from([task.task_id for task in tasks])
        G.add_edges_from([(p, task.task_id) for task in tasks for p in task.parents])
        
        # get topological sort from the graph -> dependency graph
        try:
            topo_sort = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise Exception("Scheduler Error: The provided task graph is not a DAG (contains cycles).")
        
        # assign tasks to vm in topological sort order
        vm_index = 0
        for task_id in topo_sort:
            schedule[task_id] = vm_index
            vm_index = (vm_index + 1) % num_vms # because of % it will loop back hence round first come first serve
            
        return schedule # return schedule as list of vm assignments
    

# The FCFSScheduler is "naive." It just follows its simple round-robin rule and might assign a task requiring 8192 MB of RAM to a VM with only 2048 MB.

# The main.py loop then takes this invalid schedule and passes it directly to the calculate_metrics function.

# The calculate_metrics function assumes the schedule is valid (as its docstring says) and will run the simulation anyway, producing metrics for an impossible scenario.

# This means the final results for FCFS and Round Robin might be unrealistic and overly optimistic if they happen to make invalid RAM assignments.