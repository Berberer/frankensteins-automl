import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AbstractOptimizer(ABC):
    def __init__(self, parameter_store):
        super().__init__()
        self.parameter_store = parameter_store

    @abstractmethod
    def perform_optimization(self, leaf_id, optimization_time_budget):
        pass
