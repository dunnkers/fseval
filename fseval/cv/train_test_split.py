from sklearn.model_selection import ShuffleSplit

class TrainTestSplit():
    def __init__(n_splits=10, test_size=None, train_size=None,
                 random_state=None):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state)
        self.fold = fold