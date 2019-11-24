# frankensteins-automl
**Simple usage**:
```python
from frankensteins_automl.frankensteins_automl import (
    FrankensteinsAutoMLConfig,
    FrankensteinsAutoML,
)

config = FrankensteinsAutoMLConfig("<path/to/data/arff/file>", <target_column_index>)
automl = FrankensteinsAutoML(config)
pipeline, score = automl.run()
```

**Visualize search**:
1. Add the following line to the config:
```python
config.event_send_url = "http://localhost:3000/event"
```
2. Run `yarn install` inside of the `search_visualization/` folder
3. Run `yarn electron:serve` inside of the `search_visualization/` fodler **before** you start frankensteins-automl
