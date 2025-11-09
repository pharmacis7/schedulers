from simulator.metrics import calculate_metrics
import numpy as np

class FitnessCalculator:
    """
    Calculates the fitness of a schedule for metaheuristic algorithms.
    Uses a weighted sum approach to combine multiple objectives.
    
    NEW: Now includes a check for RAM constraints.
    """
    def __init__(self, tasks, vms, weights):
        self.tasks = tasks
        self.vms = vms
        self.weights = weights
        
        # A large penalty for invalid schedules (e.g., RAM violation)
        self.max_penalty = 1_000_000.0
        
        self.max_makespan, self.max_cost, self.max_energy = self._calculate_normalization_bounds()

    def _calculate_normalization_bounds(self):
        """
        Calculates the approximate worst-case values for normalization.
        This is a rough estimate and doesn't need to be perfect.
        """
        if not self.vms:
            return 1.0, 1.0, 1.0

        slowest_vm = min(self.vms, key=lambda vm: vm.mips)
        most_expensive_vm = max(self.vms, key=lambda vm: vm.cost_per_sec)
        highest_power_vm = max(self.vms, key=lambda vm: vm.power)
        
        # Find VM with lowest (non-zero) bandwidth for worst-case
        lowest_bw_vm = min(self.vms, key=lambda vm: vm.bandwidth if vm.bandwidth > 0 else float('inf'))
        lowest_bw_mbps = (lowest_bw_vm.bandwidth / 8.0) or 0.01 # Avoid zero

        total_task_length = sum(task.length for task in self.tasks)
        
        # Estimate worst makespan: all tasks on slowest VM
        max_makespan = total_task_length / slowest_vm.mips if slowest_vm.mips > 0 else 1.0
        
        # Add a rough penalty for data transfer
        # (e.g., assume 1/4 of tasks have a data transfer on slowest link)
        total_data = sum(sum(t.parent_data_sizes.values()) for t in self.tasks)
        max_makespan += (total_data / 4.0) / lowest_bw_mbps

        max_cost = max_makespan * most_expensive_vm.cost_per_sec
        max_energy_joules = max_makespan * highest_power_vm.power
        max_energy_wh = max_energy_joules / 3600.0

        return max_makespan or 1.0, max_cost or 1.0, max_energy_wh or 1.0

    def calculate_fitness(self, schedule):
        """
        Calculates the fitness for a single schedule.
        Lower fitness is better.
        """
        
        # --- NEW: Check for Hard Constraints (RAM) ---
        # Note: This is a simple check. A more complex check would
        # track dynamic RAM usage over time, but that is far
        # more complex and not needed for this project.
        for task_id, vm_id in enumerate(schedule):
            if self.tasks[task_id].ram_required > self.vms[vm_id].ram_capacity:
                # This schedule is invalid. Return a massive penalty.
                return self.max_penalty + (self.tasks[task_id].ram_required - self.vms[vm_id].ram_capacity)

        # --- If schedule is valid, calculate metrics ---
        try:
            metrics = calculate_metrics(schedule, self.tasks, self.vms)
        except Exception as e:
            # Catch any simulation errors (like deadlocks)
            print(f"Simulation error: {e}")
            return self.max_penalty
        
        # Normalize the metrics
        norm_makespan = metrics['makespan'] / self.max_makespan
        norm_cost = metrics['total_cost'] / self.max_cost
        norm_energy = metrics['total_energy'] / self.max_energy
        
        # Weighted sum
        fitness = (
            self.weights['w_makespan'] * norm_makespan +
            self.weights['w_cost'] * norm_cost +
            self.weights['w_energy'] * norm_energy
        )
        
        return fitness