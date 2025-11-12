class Task:
    def __init__(self, task_id, length, ram_required):
        self.task_id = task_id
        self.length = length
        self.ram_required = ram_required
        
        self.parents = set()
        self.children = set()
        
        self.parent_data_sizes = {} 

    def __repr__(self):
        return (f"Task(id={self.task_id}, length={self.length}, "
                f"ram={self.ram_required}MB, parents={self.parents})")


class VM:
    def __init__(self, vm_id, mips, cost_per_sec, power, ram_capacity, bandwidth):
        self.vm_id = vm_id
        self.mips = mips
        self.cost_per_sec = cost_per_sec
        self.power = power
        self.ram_capacity = ram_capacity
        self.bandwidth = bandwidth

    def __repr__(self):
        return (f"VM(id={self.vm_id}, mips={self.mips}, ram={self.ram_capacity}MB, "
                f"bw={self.bandwidth}Mbps, cost=${self.cost_per_sec}/s)")