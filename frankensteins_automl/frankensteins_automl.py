import logging
from pubsub import pub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from frankensteins_automl.event_listener import (
    event_logger,
    event_rest_sender,
    event_topics,
)
from frankensteins_automl.machine_learning.arff_reader import read_arff
from frankensteins_automl.optimizers.evolution.genetic_algorithm import (
    GeneticAlgorithm,
)
from frankensteins_automl.optimizers.search.random_search import RandomSearch
from frankensteins_automl.mcts.mcts_search import MctsSearchConfig, MctsSearch

logger = logging.getLogger(__name__)

topic = event_topics.FRANKENSTEINS_AUTOML_TOPIC


class FrankensteinsAutoMLConfig:
    def __init__(self, data_path, data_target_column_index):
        self.data_path = data_path
        self.data_target_column_index = data_target_column_index
        self.validation_ratio = 0.2
        self.timeout_in_seconds = 300.0
        self.timout_for_optimizers_in_seconds = 30.0
        self.timeout_for_pipeline_evaluation = 10.0
        self.simulation_runs_amount = 1
        self.optimizers = [GeneticAlgorithm, RandomSearch]
        self.event_send_url = None


class FrankensteinsAutoML:
    def __init__(self, config):
        self.config = config

    def run(self):
        if self.config is None:
            logger.error(
                "No config for Frankensteins AutoML given. Cannot run."
            )
        elif not isinstance(self.config, FrankensteinsAutoMLConfig):
            logger.error(
                "Config is not a FrankensteinsAutoMLConfig. Cannot run."
            )
        else:
            # Activate event listeners
            event_logger.activate()
            if self.config.event_send_url is not None:
                event_rest_sender.activate(self.config.event_send_url)
            logger.debug("Load and split data")
            search_data, validation_data = self._load_data()
            logger.debug("Construct and init search")
            search = self._construct_search(search_data)
            logger.debug("Start search")
            pub.sendMessage(topic, payload={"event_type": "SEARCH_START"})
            pipeline, search_score = search.run_search()
            pub.sendMessage(
                topic,
                payload={
                    "event_type": "SEARCH_FINISHED",
                    "score": search_score,
                },
            )
            if pipeline is None:
                logger.warning("Was not able to find a valid pipeline in time")
                logger.warning("Please increase the given timeout")
                return None, 0.0
            else:
                logger.debug(f"Best pipeline: {pipeline}")
                logger.debug(f"Score of best pipeline: {search_score}")
                logger.debug("Validate best pipeline")
                pipeline.fit(search_data[0], search_data[1])
                predictions = pipeline.predict(validation_data[0])
                score = accuracy_score(predictions, validation_data[1])
                logger.debug(f"Validation score of best pipeline: {score}")
                return pipeline, score

    def _load_data(self):
        data_x, data_y = read_arff(
            self.config.data_path, self.config.data_target_column_index
        )
        search_x, validate_x, search_y, validate_y = train_test_split(
            data_x, data_y, test_size=self.config.validation_ratio
        )
        return (search_x, search_y), (validate_x, validate_y)

    def _construct_search(self, search_data):
        config = MctsSearchConfig(search_data[0], search_data[1])
        config.search_timeout = self.config.timeout_in_seconds
        config.optimization_time_budget = (
            self.config.timout_for_optimizers_in_seconds
        )
        config.timeout_for_pipeline_evaluation = 10.0
        config.simulation_runs_amount = self.config.simulation_runs_amount
        return MctsSearch(config, self.config.optimizers)
