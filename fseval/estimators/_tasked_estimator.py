class TaskedEstimator:
    task: Task
    classifier: Optional[Estimator]

    def __init__(
        self, task: Task, classifier: EstimatorConfig, regressor: EstimatorConfig
    ):
        ...
