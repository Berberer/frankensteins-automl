import logging
from frankensteins_automl.search_space import search_space_reader
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceGraphGenerator,
)

logger = logging.getLogger("test-log")
logger.addHandler(logging.NullHandler())


class TestSearchSpaceGraph:
    def test_search_space_enumeration(self):
        search_space = search_space_reader.create_search_space(
            "res/ml-plan-ul.json",
            "res/scikit-learn-classifiers-tpot.json",
            "res/scikit-learn-preprocessors-tpot.json",
        )
        generator = SearchSpaceGraphGenerator(
            search_space, "sklearn.pipeline.make_pipeline"
        )
        root = generator.get_root_node()
        open_list = [root]
        leaf_nodes = []
        while len(open_list) > 0:
            logger.info(f"Started step with open length {len(open_list)}")
            node = open_list[0]
            open_list = open_list[1:]
            sucessors = generator.generate_sucessors(node)
            if len(sucessors) > 0:
                logger.info(f"Adding {len(sucessors)} new nodes")
                open_list.extend(sucessors)
            else:
                logger.info("Found a leaf node")
                leaf_nodes.append(node)
        for leaf in leaf_nodes:
            logger.info(leaf.get_rest_problem().get_component_mapping())
            logger.info(leaf.get_rest_problem().get_required_interfaces())
            logger.info("################################")
        assert len(leaf_nodes) > 0
