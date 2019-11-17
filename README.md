# frankensteins-automl
Simple usage:
```python
from frankensteins_automl.frankensteins_automl import (
    FrankensteinsAutoMLConfig,
    FrankensteinsAutoML,
)

config = FrankensteinsAutoMLConfig("<path/to/data/arff/file>", <target_column_index>)
automl = FrankensteinsAutoML(config)
pipeline, score = automl.run()
```
