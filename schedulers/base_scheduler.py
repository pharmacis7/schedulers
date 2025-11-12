from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    
    @abstractmethod
    def schedule(self, tasks, vms):
        pass
    
    def set_fitness_weights(self, weights, tasks, vms):
        pass