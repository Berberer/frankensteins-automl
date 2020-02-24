import logging
from smac.configspace import ConfigurationSpace, UniformFloatHyperparameter
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)


logger = logging.getLogger(__name__)


class GeneticAlgorithm(AbstractOptimizer):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        timeout_for_pipeline_evaluation,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
        )

    def _create_configuration_space(self):
        config_space = ConfigurationSpace()
        min, max = self.parameter_domain._calculate_min_and_max_vectors()
        for i in range(len(min)):
            config_space.add_hyperparameter(
                UniformFloatHyperparameter(
                    name=str(i), lower=min[i], upper=max[i]
                )
            )

    def perform_optimization(self, optimization_time_budget):
        pass
