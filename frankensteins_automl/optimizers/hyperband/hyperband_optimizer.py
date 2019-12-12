import logging
from time import time
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)
from frankensteins_automl.optimizers.hyperband.hyperband_run import (
    HyperbandRun,
)

logger = logging.getLogger(__name__)


class Hyperband(AbstractOptimizer):
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
        self.hyperband_run = HyperbandRun(
            self._select_hyperband_parameters, super()._score_candidate
        )
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self._score_candidate(self.best_candidate)

    def perform_optimization(self, optimization_time_budget):
        spend_time_budget = 0
        last_hyperband_run = 0
        while (
            spend_time_budget + last_hyperband_run
        ) <= optimization_time_budget:
            run_start = time()
            run_results = self.hyperband_run.run()
            self._filter_hyperband_results(run_results)
            last_hyperband_run = time() - run_start
            spend_time_budget = spend_time_budget + last_hyperband_run
        return (
            self.parameter_domain.config_from_vector(self.best_candidate),
            self.best_score,
        )

    def _filter_hyperband_results(self, results):
        # TODO
        pass

    def _select_hyperband_candidates(self):
        # TODO
        pass
