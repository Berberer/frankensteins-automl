import arff
import json
import numpy
import logging
from frankensteins_automl.machine_learning.pipelines import (
    pipeline_evaluator,
    pipeline_constructor,
)
from frankensteins_automl.optimizers.optimization_parameter_domain import (
    OptimizationParameterDomain,
)
from frankensteins_automl.search_space import search_space_reader
from frankensteins_automl.search_space.search_space_graph import (
    SearchSpaceGraphGenerator,
)
from sklearn.pipeline import Pipeline

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
while len(open_list) > 0 and len(lengths) < 3:
    node = open_list[0]
    open_list = open_list[1:]
    successors = generator.generate_successors(node)
    if len(successors) > 0:
        open_list.extend(successors)
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

    def test_leaf_node_pipeline_creation(self):
        data = numpy.array(
            arff.load(open("res/datasets/blood_transfusion.arff", "r"))["data"]
        )
        data = data.astype(numpy.float64)
        data_x = data[:, :4]
        data_y = data[:, 4]
        for leaf in leaf_nodes:
            logger.info("Component mapping: ")
            cm = leaf.get_rest_problem().get_component_mapping()
            param_domain = OptimizationParameterDomain(cm)
            default_params = param_domain.get_default_config()
            for id, component in cm.items():
                logger.info(f"\t{id}:{component.get_name()}")
            logger.info(f"\tParams: {default_params}")
            evaluator = pipeline_evaluator.PipelineEvaluator(
                numpy.copy(data_x),
                numpy.copy(data_y),
                "sklearn.pipeline.make_pipeline",
                leaf.get_rest_problem(),
            )
            pipeline = pipeline_constructor.construct_pipeline(
                "sklearn.pipeline.make_pipeline",
                leaf.get_rest_problem(),
                default_params,
            )
            assert pipeline is not None
            assert isinstance(pipeline, Pipeline)
            try:
                score = evaluator.evaluate_pipeline(default_params)
                logger.info(f"\tScore: {score}")
                assert score > 0.5
                logger.info("################################")
            except Exception as e:
                logger.exception(f"Error while scoring pipeline: {e}")
