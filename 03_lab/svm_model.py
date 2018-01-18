import pickle

from hyperopt import hp
from sklearn.svm import SVC

from modelgym.models import Model, LearningTask
from modelgym.utils.dataset import XYCDataset
from hyperopt import hp
import numpy as np


class SVCClassifier(Model):
    def __init__(self, params=None):
        """
        Args:
            params (dict or None): parameters for model.
                             If None default params are fetched.
            learning_task (str): set type of task(classification, regression, ...)
        """

        if params is None:
            params = {}

        self.params = {    
            'C':1.0, 
            'kernel':'rbf',
            'degree':3,
            'gamma':'auto',
            'verbose':0
        }
        if params:
            self.params.update(params)

        self.model = None

    def _set_model(self, model):
        """
        sets new model, internal method, do not use
        Args:
            model: internal model
        """
        self.model = model

    def _convert_to_dataset(self, data, label, cat_cols=None):
        return XYCDataset(data, label, cat_cols)

    def fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
        Return:
            self
        """
        dtrain = self._convert_to_dataset(dataset.X, dataset.y)
        self.model = SVC(C=self.params['C'], kernel=self.params['kernel'],
                         degree=self.params['degree'], gamma=self.params['gamma'],
                         verbose=self.params['verbose']).fit(dtrain.X, dtrain.y)
        return self

    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.

        """
        assert self.model is not None, "model is not fitted"
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        new_model = SVCClassifier(model.get_params())
        new_model._set_model(model)
        return new_model

    def predict(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, ) or (n_samples, n_outputs)
        """
        return self.model.predict(dataset.X)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False

    def predict_proba(self, dataset):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        return self.model.predict_proba(dataset.X)[:, 1]

    @staticmethod
    def get_default_parameter_space():
        """
        Return:
            dict of DistributionWrappers
        """

        return {
            'C':      hp.choice('C', np.logspace(-5, 5, 10)), 
            'kernel': hp.choice('kernel', ['rbf']),
            #'kernel': hp.choice('kernel', ['rbf']),
            #'degree': hp.choice('degree', range(1, 10)),
            'gamma':  hp.choice('gamma', np.logspace(-5,5,10))
        }

    @staticmethod
    def get_learning_task():
        return LearningTask.CLASSIFICATION
