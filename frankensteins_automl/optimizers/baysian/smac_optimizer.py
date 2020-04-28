import logging
import numpy
from smac.configspace import (
    Configuration,
    ConfigurationSpace,
    UniformFloatHyperparameter,
)
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.objective import average_cost
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType


logger = logging.getLogger(__name__)


class SMAC(AbstractOptimizer):
    def __init__(
        self,
        parameter_domain,
        pipeline_evaluator,
        timeout_for_pipeline_evaluation,
        numpy_random_state,
    ):
        super().__init__(
            parameter_domain,
            pipeline_evaluator,
            timeout_for_pipeline_evaluation,
            numpy_random_state,
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)
        self.configuration_space = self._create_configuration_space()
        self.candidates_for_warmstart_history = 100

    def _create_configuration_space(self):
        config_space = ConfigurationSpace()
        min, max = self.parameter_domain._calculate_min_and_max_vectors()
        for i in range(len(min)):
            config_space.add_hyperparameter(
                UniformFloatHyperparameter(
                    name=str(i), lower=min[i], upper=max[i]
                )
            )
        return config_space

    def _create_scenario(self, optimization_time_budget):
        return Scenario(
            {
                "cs": self.configuration_space,
                "run_obj": "quality",
                "wallclock_limit": optimization_time_budget,
                "deterministic": "true",
            }
        )

    def _create_run_history(self):
        runhistory = RunHistory(aggregate_func=average_cost)
        candidates = []
        candidates.extend(
            self.parameter_domain.get_top_results(
                int(self.candidates_for_warmstart_history * 0.5)
            )
        )
        if len(candidates) == int(self.candidates_for_warmstart_history * 0.5):
            for _ in range(int(self.candidates_for_warmstart_history * 0.5)):
                candidates.append(self.parameter_domain.get_random_result())
        for score, candidate in candidates:
            runhistory.add(
                config=Configuration(
                    self.configuration_space,
                    values=self._vector_to_smac_config(candidate),
                ),
                cost=(1 - score),
                time=self.pipeline_evaluation_timeout,
                status=StatusType.SUCCESS,
            )
        return runhistory

    def _smac_config_to_vector(self, config):
        length = len(config)
        vector = numpy.zeros(length)
        for i in range(length):
            vector[i] = config[str(i)]
        return vector

    def _vector_to_smac_config(self, vector):
        length = len(vector)
        config = {}
        for i in range(length):
            config[str(i)] = vector[i]
        return config

    def perform_optimization(self, optimization_time_budget):
        def _evaluate_config(config):
            score = 0.5
            return 1 - score

        smac = SMAC4HPO(
            scenario=self._create_scenario(
                optimization_time_budget - self.pipeline_evaluation_timeout
            ),
            tae_runner=_evaluate_config,
            rng=self.numpy_random_state,
            runhistory=self._create_run_history(),
        )
        try:
            candidate = smac.optimize()
        finally:
            candidate = smac.solver.incumbent
        score = _evaluate_config(candidate)
        if score > self.best_score:
            self.best_score = score
            self.best_candidate = self._smac_config_to_vector(candidate)
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )
