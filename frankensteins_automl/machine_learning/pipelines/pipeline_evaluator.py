import logging
from frankensteins_automl.machine_learning.pipelines import (
    pipeline_constructor,
)
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    def __init__(
        self, data_x, data_y, start_component_name, satisfied_rest_problem
    ):
        self.data_x = data_x
        self.data_y = data_y
        self.start_component_name = start_component_name
        self.rest_problem = satisfied_rest_problem

    def evaluate_pipeline(self, pipeline_parameter_config):
        score = 0.0
        try:
            pipeline = pipeline_constructor.construct_pipeline(
                self.start_component_name,
                self.rest_problem,
                pipeline_parameter_config,
            )
            if pipeline is not None:
                score = cross_val_score(
                    pipeline,
                    self.data_x,
                    self.data_y,
                    cv=10,
                    error_score="raise",
                ).mean()
                logger.debug(f"{pipeline} achieved : {score}")
            else:
                logger.warning("Constructed pipeline is None")
        except Exception as e:
            logger.exception(f"Error while scoring pipeline: {e}")
        return score
