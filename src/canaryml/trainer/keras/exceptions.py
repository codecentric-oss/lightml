class NotAKerasModelError(Exception):
    def __init__(self, msg: str = "The model is not a Keras model."):
        super().__init__(msg)


class UnknownOptimizerError(Exception):
    def __init__(
        self,
        msg: str = "The optimizer you provided is not recognized. "
        "Please choose a valid optimizer from Keras.",
    ):
        super().__init__(msg)


class UnknownLossFunctionError(Exception):
    def __init__(
        self,
        msg: str = "The loss function you provided is not recognized. "
        "Please choose a valid loss function from Keras.",
    ):
        super().__init__(msg)


class MissingMetricNameError(Exception):
    def __init__(
        self,
        msg: str = "A provided metric entry does not have a metric name ('metric_name'). "
        "Please check the configuration of the metrics to be used. ",
    ):
        super().__init__(msg)


class UnknownMetricError(Exception):
    def __init__(
        self,
        msg: str = "The metric name you provided is not recognized. "
        "Please choose a valid metric name from Keras.",
    ):
        super().__init__(msg)
