from frankensteins_automl.search_space import search_space_reader
from frankensteins_automl.search_space.search_space import SearchSpace


class TestSearchSpaceReader:
    def test_non_string_path(self):
        assert search_space_reader.create_search_space(123) is None

    def test_non_json_path(self):
        assert search_space_reader.create_search_space("README.md") is None

    def test_json_parsing(self):
        components = search_space_reader.create_search_space(
            "res/ml-plan-ul.json"
        )
        assert isinstance(components, SearchSpace)
