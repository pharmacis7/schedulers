import networkx as nx
from schedulers.base_scheduler import BaseScheduler

class RRScheduler(BaseScheduler):
    """
    Round Robin (RR) Scheduler for DAGs.
    
    In a static batch environment, this behaves identically
    to the FCFS (DAG) interpretation: it cycles through the 
    VM list for tasks in a topological order.
    """

    def schedule(self, tasks, vms):
        num_vms = len(vms)
        num_tasks = len(tasks)
        schedule = [0] * num_tasks
        
        # build fresh graph
        G = nx.DiGraph()
        G.add_nodes_from([task.task_id for task in tasks])
        G.add_edges_from([(p, task.task_id) for task in tasks for p in task.parents])
        
        # get topological order
        try:
            topo_sort = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise Exception("Scheduler Error: The provided task graph is not a DAG (contains cycles).")
        
        # we cant assign time slice because one task depend on other -> it becomes most likely fcfs
        vm_index = 0
        for task_id in topo_sort:
            schedule[task_id] = vm_index
            vm_index = (vm_index + 1) % num_vms
            
        return schedule