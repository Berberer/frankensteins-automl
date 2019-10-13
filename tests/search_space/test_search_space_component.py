from frankensteins_automl.search_space.search_space_component import (
    SearchSpaceComponent,
)


component_description = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "requiredInterface": [
        {"id": "required1", "name": "requiredA"},
        {"id": "required2", "name": "requiredB"},
        {"id": "required3", "name": "requiredC"},
    ],
    "parameter": [
        {
            "name": "testDouble",
            "type": "double",
            "default": 0.53,
            "min": 0.05,
            "max": 1.01,
            "minInterval": 0.05,
            "refineSplits": 2,
        },
        {
            "name": "testInt",
            "type": "int",
            "default": 6,
            "min": 1,
            "max": 11,
            "minInterval": 1,
            "refineSplits": 2,
        },
        {
            "name": "testCat",
            "default": "a",
            "type": "cat",
            "values": ["a", "b", "c"],
        },
    ],
}


component_description_with_unknown_type = {
    "name": "testComponent",
    "providedInterface": ["providedA", "providedB", "providedC"],
    "requiredInterface": [{"id": "required1", "name": "requiredA"}],
    "parameter": [
        {"name": "testAbc", "type": "abc"},
        {
            "name": "testCat",
            "default": "a",
            "type": "cat",
            "values": ["a", "b", "c"],
        },
    ],
}


class TestSearchSpaceComponent:
    def test_search_space_component_creation(self):
        component = SearchSpaceComponent(component_description)
        assert component.get_name() == "testComponent"
        assert component.get_provided_interfaces() == [
            "providedA",
            "providedB",
            "providedC",
        ]
        assert component.get_required_interfaces() == [
            {"id": "required1", "name": "requiredA"},
            {"id": "required2", "name": "requiredB"},
            {"id": "required3", "name": "requiredC"},
        ]

    def test_correct_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "testCat": "b"}
        )

    def test_unknown_category_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "testCat": "d"}
        )

    def test_too_small_value_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 0.04, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 0, "testCat": "d"}
        )

    def test_too_big_value_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 1.02, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 12, "testCat": "d"}
        )

    def test_wrong_type_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 1, "testInt": 5, "testCat": "d"}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5.5, "testCat": "d"}
        )

    def test_missing_param_configuration_validation(self):
        component = SearchSpaceComponent(component_description)
        assert not component.validate_parameter_config(
            {"testDouble": 1, "testInt": 5}
        )
        assert not component.validate_parameter_config(
            {"testDouble": 0.5, "testInt": 5, "anotherCat": "d"}
        )

    def test_unkwon_type_param_configuration_validation(self):
        component = SearchSpaceComponent(
            component_description_with_unknown_type
        )
        assert component.validate_parameter_config(
            {"testAbc": 1, "testCat": "a"}
        )
