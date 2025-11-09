from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    
    # abstract base class for all scheduling algos
    @abstractmethod
    def schedule(self, tasks, vms):
        pass
    
    def set_fitness_weights(self, weights, tasks, vms):
        pass