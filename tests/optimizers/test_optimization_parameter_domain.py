from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)

component_mapping = {
    "abc": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [
                {
                    "id": "required1",
                    "name": "requiredA",
                    "construction_key": "key_a",
                },
                {
                    "id": "required2",
                    "name": "requiredB",
                    "construction_key": 2,
                },
                {
                    "id": "required3",
                    "name": "requiredC",
                    "construction_key": "key_b",
                },
            ],
            "parameter": [
                {
                    "name": "testDouble",
                    "type": "double",
                    "default": 0.53,
                    "min": 0.05,
                    "max": 1.01,
                    "construction_key": 1,
                },
                {
                    "name": "testInt",
                    "type": "int",
                    "default": 6,
                    "min": 1,
                    "max": 11,
                    "construction_key": 0,
                },
                {
                    "name": "testCat",
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                    "construction_key": "key_c",
                },
            ],
        }
    ),
    "def": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [{"id": "required1", "name": "requiredA"}],
            "parameter": [
                {
                    "name": "testCat",
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                }
            ],
        }
    ),
    "ghi": SearchSpaceComponent(
        {
            "name": "testComponent",
            "providedInterface": ["providedA", "providedB", "providedC"],
            "requiredInterface": [{"id": "required1", "name": "requiredA"}],
            "parameter": [],
        }
    ),
}


class TestOptimizationParameterDomain:
    def test_default_configuration(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert domain.get_default_config() == {
            "abc": {"testDouble": 0.53, "testInt": 6, "testCat": "a"},
            "def": {"testCat": "a"},
            "ghi": {},
        }

    def test_random_configuration_completeness(self):
        domain = OptimizationParameterDomain(component_mapping)
        random_config = domain.draw_random_config()
        assert "abc" in random_config
        assert "testDouble" in random_config["abc"]
        assert "testInt" in random_config["abc"]
        assert "testCat" in random_config["abc"]
        assert "def" in random_config
        assert "testCat" in random_config["def"]
        assert random_config["ghi"] == {}

    def test_parameter_descriptions(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert domain.get_parameter_descriptions() == {
            "abc": {
                "testDouble": {
                    "type": "double",
                    "default": 0.53,
                    "min": 0.05,
                    "max": 1.01,
                    "construction_key": 1,
                },
                "testInt": {
                    "type": "int",
                    "default": 6,
                    "min": 1,
                    "max": 11,
                    "construction_key": 0,
                },
                "testCat": {
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                    "construction_key": "key_c",
                },
            },
            "def": {
                "testCat": {
                    "default": "a",
                    "type": "cat",
                    "values": ["a", "b", "c"],
                }
            },
            "ghi": {},
        }

    def test_queuing_order(self):
        domain = OptimizationParameterDomain(component_mapping)
        assert not domain.has_results()
        domain.add_result({"abc": 123}, 12)
        domain.add_result({"def": 456}, 3)
        domain.add_result({"ghi": 789}, 42)
        assert domain.has_results()
        assert domain.get_top_results(1) == [(42, {"ghi": 789})]
        assert domain.get_top_results(3) == [
            (42, {"ghi": 789}),
            (12, {"abc": 123}),
            (3, {"def": 456}),
        ]
