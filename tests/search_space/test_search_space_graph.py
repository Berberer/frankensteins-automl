import json
import logging
from frankensteins_automl.search_space import search_space_reader
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceGraphGenerator,
)

logger = logging.getLogger("test-log")
logger.addHandler(logging.NullHandler())


# Generate a big enoug search space graph
search_space = search_space_reader.create_search_space(
    "res/search_space/ml-plan-ul.json",
    "res/search_space/scikit-learn-classifiers-tpot.json",
    "res/search_space/scikit-learn-preprocessors-tpot.json",
)
generator = SearchSpaceGraphGenerator(
    search_space, "sklearn.pipeline.make_pipeline"
)
root = generator.get_root_node()
open_list = [root]
leaf_nodes = []
lengths = []
while len(open_list) > 0 and len(leaf_nodes) < 4:
    node = open_list[0]
    open_list = open_list[1:]
    sucessors = generator.generate_sucessors(node)
    if len(sucessors) > 0:
        open_list.extend(sucessors)
    else:
        length = len(node.get_rest_problem().get_required_interfaces())
        if length not in lengths:
            lengths.append(length)
            leaf_nodes.append(node)


class TestSearchSpaceGraph:
    def test_search_space_interfaces_satisfied_in_leaf(self):
        logger.info("Results of search space enumeration: ")
        for leaf in leaf_nodes:
            cm = leaf.get_rest_problem().get_component_mapping()
            ri = leaf.get_rest_problem().get_required_interfaces()
            logger.info(f"Required Interfaces:\n{json.dumps(ri, indent=4)}")
            logger.info("Component mapping: ")
            for id, component in cm.items():
                logger.info(f"\t{id}:{component.get_name()}")
            logger.info("################################")
            for interface in ri:
                assert interface["satisfied"]
        assert len(leaf_nodes) > 0
