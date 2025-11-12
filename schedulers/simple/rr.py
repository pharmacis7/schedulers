import networkx as nx
from schedulers.base_scheduler import BaseScheduler

class RRScheduler(BaseScheduler):

    def schedule(self, tasks, vms):
        num_vms = len(vms)
        num_tasks = len(tasks)
        schedule = [0] * num_tasks
        
        G = nx.DiGraph()
        G.add_nodes_from([task.task_id for task in tasks])
        G.add_edges_from([(p, task.task_id) for task in tasks for p in task.parents])
        
        try:
            topo_sort = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise Exception("Scheduler Error: The provided task graph is not a DAG (contains cycles).")
        
        vm_index = 0
        for task_id in topo_sort:
            schedule[task_id] = vm_index
            vm_index = (vm_index + 1) % num_vms
            
        return schedule