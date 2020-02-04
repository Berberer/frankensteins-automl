from frankensteins_automl.frankensteins_automl import (
    FrankensteinsAutoML,
    FrankensteinsAutoMLConfig,
)

config = FrankensteinsAutoMLConfig("res/datasets/blood_transfusion.arff", 4)
config.timeout_in_seconds = 180
config.timout_for_optimizers_in_seconds = 45.0
config.timeout_for_pipeline_evaluation = 15.0
automl = FrankensteinsAutoML(config)
pipeline, score = automl.run()

print(pipeline)
print(score)
