import numpy as np

def calculate_metrics(schedule, tasks, vms):
    
    num_tasks = len(tasks)
    num_vms = len(vms)

    vm_finish_times = np.zeros(num_vms)
    
    vm_total_busy_time = np.zeros(num_vms)
    
    task_finish_times = {} 

    uncompleted_parents_count = {
        task.task_id: len(task.parents) for task in tasks
    }

    ready_queue = []
    for task in tasks:
        if uncompleted_parents_count[task.task_id] == 0:
            ready_queue.append(task.task_id)

    while ready_queue or len(task_finish_times) < num_tasks:
        if not ready_queue:
            raise Exception("Error in DAG simulation: Deadlock or cycle detected.")
            
        task_id = ready_queue.pop(0)
        task = tasks[task_id]
        
        vm_id = schedule[task_id]
        vm = vms[vm_id]

        vm_available_time = vm_finish_times[vm_id]
        
        parent_data_ready_time = 0
        if task.parents:
            for p_id in task.parents:
                p_finish_time = task_finish_times[p_id]
                p_vm_id = schedule[p_id]
                
                transfer_time = 0
                if p_vm_id != vm_id:
                    data_size_mb = task.parent_data_sizes[p_id]
                    bandwidth_mbps = vm.bandwidth / 8.0 
                    if bandwidth_mbps > 0:
                        transfer_time = data_size_mb / bandwidth_mbps
                    else:
                        transfer_time = float('inf')
                
                this_parent_ready_time = p_finish_time + transfer_time
                
                if this_parent_ready_time > parent_data_ready_time:
                    parent_data_ready_time = this_parent_ready_time
            
        start_time = max(vm_available_time, parent_data_ready_time)
        
        execution_time = task.length / vm.mips
        finish_time = start_time + execution_time
        
        task_finish_times[task_id] = finish_time
        vm_finish_times[vm_id] = finish_time
        vm_total_busy_time[vm_id] += execution_time
        
        for child_id in task.children:
            uncompleted_parents_count[child_id] -= 1
            if uncompleted_parents_count[child_id] == 0:
                ready_queue.append(child_id)

    makespan = max(task_finish_times.values())
    
    total_cost = 0
    for vm_id in range(num_vms):
        total_cost += vm_total_busy_time[vm_id] * vms[vm_id].cost_per_sec
        
    total_energy_joules = 0
    for vm_id in range(num_vms):
        total_energy_joules += vm_total_busy_time[vm_id] * vms[vm_id].power
    total_energy_wh = total_energy_joules / 3600.0
    
    return {
        "makespan": makespan,
        "total_cost": total_cost,
        "total_energy": total_energy_wh
    }