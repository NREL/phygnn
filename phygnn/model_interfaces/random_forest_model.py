# -*- coding: utf-8 -*-
"""
Random Forest Model
"""
import json
import logging
import os
from sklearn.ensemble import RandomForestRegressor

from phygnn.model_interfaces.base_model import ModelBase

logger = logging.getLogger(__name__)


class RandomForestModel(ModelBase):
    """
    scikit learn Random Forest Regression
    """

    def __init__(self, model, feature_names=None, label_name=None,
                 norm_params=None):
        """
        Parameters
        ----------
        model : sklearn.ensemble.RandomForestRegressor
            Sklearn Random Forest Model
        feature_names : list
            Ordered list of feature names.
        label_name : str
            label (output) variable name.
        norm_params : dict, optional
            Dictionary mapping feature and label names (keys) to normalization
            parameters (mean, stdev), by default None
        """
        super().__init__(model, feature_names=feature_names,
                         label_names=label_name, norm_params=norm_params)

        if len(self.label_names) > 1:
            msg = ("Only a single label can be supplied to {}, but {} were"
                   .format(self.__class__.__name__, len(self.label_names)))
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def compile_model(**kwargs):
        """
        Build sklearn random forest model

        Parameters
        ----------
        kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor

        Returns
        -------
        sklearn.ensemble.RandomForestRegressor
            sklearn random forest model
        """
        model = RandomForestRegressor(**kwargs)

        return model

    def unnormalize_prediction(self, prediction):
        """
        Unnormalize prediction if needed

        Parameters
        ----------
        prediction : ndarray
           Model prediction

        Returns
        -------
        prediction : ndarray
            Native prediction
        """
        means = self.label_means[0]
        if means:
            stdevs = self.label_stdevs[0]
            prediction = self._unnormalize(prediction, means, stdevs)

        return prediction

    def _parse_labels(self, label, name=None, normalize=True):
        """
        Parse labels and normalize if desired

        Parameters
        ----------
        label : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        name : list, optional
            List of label names, by default None
        normalize : bool, optional
            Normalize label array, by default True

        Returns
        -------
        label : ndarray
            Parsed labels array, normalized if desired
        """
        label = super()._parse_labels(label, names=name,
                                      normalize=normalize)

        if len(self.label_names) > 1:
            msg = ("Only a single label can be supplied to {}, but {} were"
                   .format(self.__class__.__name__, len(self.label_names)))
            logger.error(msg)
            raise ValueError(msg)

        return label

    def train_model(self, features, label, norm_label=True, parse_kwargs=None,
                    fit_kwargs=None):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Input features to train on
        label : dict | pandas.DataFrame
            label to train on
        norm_label : bool
            Flag to normalize label
        parse_kwargs : dict
            kwargs for cls._parse_features
        fit_kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor.fit
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        features = self._parse_features(features, **parse_kwargs)

        label = self._parse_labels(label, normalize=norm_label)

        if fit_kwargs is None:
            fit_kwargs = {}

        # pylint: disable=no-member
        self._model.fit(features, label.ravel(), **fit_kwargs)

    def save_model(self, path):
        """
        Save Random Forest Model to path.

        Parameters
        ----------
        path : str
            Path to save model to
        """
        if path.endswith('.json'):
            dir_path = os.path.dirname(path)
        else:
            dir_path = path
            path = os.path.join(dir_path, os.path.basename(path) + '.json')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'model_params': self.model.get_params()}

        model_params = self.dict_json_convert(model_params)
        with open(path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

    @classmethod
    def train(cls, features, label, norm_label=True, save_path=None,
              compile_kwargs=None, parse_kwargs=None, fit_kwargs=None):
        """
        Build Random Forest Model with given kwargs and then train with
        given features, labels, and kwargs

        Parameters
        ----------
        features : pandas.DataFrame
            Model features
        label : pandas.DataFrame
            label to train on
        norm_label : bool
            Flag to normalize label
        save_path : str
            Directory path to save model to. The RandomForest Model will be
            saved to the directory while the framework parameters will be
            saved in json.
        compile_kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor
        parse_kwargs : dict
            kwargs for cls._parse_features
        fit_kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor.fit

        Returns
        -------
        model : RandomForestModel
            Initialized and trained RandomForestModel obj
        """
        if compile_kwargs is None:
            compile_kwargs = {}

        _, feature_names = cls._parse_data(features)
        _, label_name = cls._parse_data(label)

        model = cls(cls.compile_model(**compile_kwargs),
                    feature_names=feature_names, label_name=label_name)

        model.train_model(features, label, norm_label=norm_label,
                          parse_kwargs=parse_kwargs, fit_kwargs=fit_kwargs)

        if save_path is not None:
            model.save_model(save_path)

        return model

    @classmethod
    def load(cls, path):
        """
        Load model from model path.

        Parameters
        ----------
        path : str
            Directory path to TfModel to load model from. There should be a
            tensorflow saved model directory with a parallel pickle file for
            the TfModel framework.

        Returns
        -------
        model : TfModel
            Loaded TfModel from disk.
        """
        if not path.endswith('.json'):
            path = os.path.join(path, os.path.basename(path) + '.json')

        if not os.path.exists(path):
            e = ('{} does not exist'.format(path))
            logger.error(e)
            raise IOError(e)

        with open(path, 'r') as f:
            model_params = json.load(f)

        loaded = RandomForestRegressor()
        loaded = loaded.set_params(**model_params.pop('model_params'))

        model = cls(loaded, **model_params)

        return model
