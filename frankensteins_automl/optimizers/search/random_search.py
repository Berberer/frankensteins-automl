import asyncio
import copy
import logging
import random
from frankensteins_automl.optimizers.abstract_optimizer import (
    AbstractOptimizer,
)

logger = logging.getLogger(__name__)


class RandomSearch(AbstractOptimizer):
    def __init__(self, parameter_domain, pipeline_evaluator):
        super().__init__(parameter_domain, pipeline_evaluator)
        self.best_candidate = self.parameter_domain.get_default_config()
        self.best_score = self.pipeline_evaluator.evaluate_parameter(
            self.best_candidate
        )

    async def perform_optimization(self, optimization_time_budget):
        if self.parameter_domain.has_results():
            best_from_domain = self.parameter_domain.get_top_results(1)
            self.best_score, self.best_candidate = best_from_domain
        logger.info(f"Random Search starts with score: {self.best_score}")
        try:
            await asyncio.wait_for(
                self._search_awaitable(), timeout=optimization_time_budget
            )
        except asyncio.TimeoutError:
            logger.info(f"Optimization timeout.")
        logger.info(f"Random Search ends with score: {self.best_score}")
        return self.best_candidate, self.best_score

    async def _search_loop(self):
        while True:
            candidate = self._next_step()
            candidate_score = self.pipeline_evaluator.evaluate_parameter(
                candidate
            )
            self.parameter_domain.add_result(candidate, candidate_score)
            logger.info(
                f"Random search found a config with score: {candidate_score}"
            )
            if candidate_score > self.best_score:
                self.best_candidate = candidate
                self.best_score = candidate_score

    def _next_step(self):
        candidate = copy.deepcopy(self.best_candidate)
        # Select a random parameter from a random component to change
        changed_component = candidate[random.choice(list(candidate.keys()))]
        changed_parameter = random.choice(
            list(candidate[changed_component].keys())
        )
        param_description = self.parameter_domain.get_parameter_descriptions()[
            changed_component
        ][changed_parameter]
        current_value = candidate[changed_component][changed_parameter]
        new_value = current_value
        if param_description["type"] == "int":
            # For an int, select a neighbouring int
            if current_value == param_description["min"]:
                new_value = new_value + 1
            elif current_value == param_description["max"]:
                new_value = new_value - 1
            else:
                new_value = new_value + random.choice([-1, 1])
        elif param_description["type"] == "double":
            # For a double d, select a value from [d-0.1, d+0.1]
            lower_bound = new_value
            upper_bound = new_value
            if (lower_bound - 0.1) >= param_description["min"]:
                lower_bound = lower_bound - 0.1
            if (upper_bound + 0.1) <= param_description["min"]:
                upper_bound = upper_bound + 0.1
            new_value = random.uniform(lower_bound, upper_bound)
        elif param_description["type"] == "cat":
            # For a cat, select a neighbouring cat value
            index = param_description["values"].index(new_value)
            if index == 0:
                new_value = param_description["values"][1]
            elif index == (len(param_description["values"]) - 1):
                new_value = param_description["values"][-2]
            else:
                new_index = index + random.choice([-1, 1])
                new_value = param_description["values"][new_index]
        elif param_description["type"] == "bool":
            # For a bool, select the opposite
            if new_value:
                new_value = False
            else:
                new_value = True
        candidate[changed_component][changed_parameter] = new_value
        return candidate
