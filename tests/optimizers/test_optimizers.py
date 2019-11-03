from time import process_time
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.optimizers.search.random_search import RandomSearch
from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)

component_mapping = {
    "vars": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": [],
            "requiredInterface": [],
            "parameter": [
                {
                    "name": "x",
                    "type": "double",
                    "default": 2.0,
                    "min": -5.0,
                    "max": 5.0,
                    "construction_key": 1,
                },
                {
                    "name": "y",
                    "type": "double",
                    "default": -2.0,
                    "min": -5.0,
                    "max": 5.0,
                    "construction_key": 2,
                },
            ],
        }
    )
}

optimizer_classes = [RandomSearch]


class TestEvaluator:
    def evaluate_pipeline(self, params):
        return -(params["vars"]["x"]) ** 2 - (params["vars"]["y"]) ** 2


class TestOptimizers:
    def test_optimization_result_improvements(self):
        domain = OptimizationParameterDomain(component_mapping)
        for optimizer_class in optimizer_classes:
            optimizer = optimizer_class(domain, TestEvaluator())
            _, score = optimizer.perform_optimization(10)
            print(score)
            assert score > -8.0

    def test_optimization_timeout(self):
        timeout_in_seconds = 5
        domain = OptimizationParameterDomain(component_mapping)
        for optimizer_class in optimizer_classes:
            optimizer = optimizer_class(domain, TestEvaluator())
            start_time = process_time()
            optimizer.perform_optimization(timeout_in_seconds)
            stop_time = process_time()
            assert (stop_time - start_time) < (timeout_in_seconds + 1)
            assert (stop_time - start_time) > (timeout_in_seconds - 1)
