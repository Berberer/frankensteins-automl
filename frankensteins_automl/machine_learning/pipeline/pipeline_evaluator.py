import logging
import stopit
import warnings
from frankensteins_automl.machine_learning.pipeline import pipeline_constructor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    def __init__(
        self, data_x, data_y, start_component_name, satisfied_rest_problem
    ):
        self.data_x = data_x
        self.data_y = data_y
        self.start_component_name = start_component_name
        self.rest_problem = satisfied_rest_problem

    @stopit.threading_timeoutable(default=0.0)
    def evaluate_pipeline(self, pipeline_parameter_config, ratio=1.0):
        score = 0.0
        try:
            pipeline = pipeline_constructor.construct_pipeline(
                self.start_component_name,
                self.rest_problem,
                pipeline_parameter_config,
            )
            if pipeline is not None:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    data_x = self.data_x
                    data_y = self.data_y
                    # Try to create a subset of the training data
                    # with the given ratio
                    try:
                        if ratio < 1.0:
                            data_x, _, data_y, _ = train_test_split(
                                data_x, data_y, train_size=ratio
                            )
                    finally:
                        # Score the pipeline with the cross-validation
                        # mean on the data or the sample split
                        score = cross_val_score(
                            pipeline,
                            data_x,
                            data_y,
                            cv=10,
                            error_score="raise",
                        ).mean()
                        logger.debug(f"{pipeline} achieved : {score}")
                    for warning in w:
                        logger.debug(f"Pipeline evaluation warning: {warning}")
            else:
                logger.warning("Constructed pipeline is None")
        except Exception as e:
            logger.exception(f"Error while scoring pipeline: {e}")
        return score
