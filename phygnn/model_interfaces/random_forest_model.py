# -*- coding: utf-8 -*-
"""
Random Forest Model
"""
import json
import logging
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

from phygnn.model_interfaces.base_model import ModelBase
from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


class RandomForestModel(ModelBase):
    """
    scikit learn Random Forest Regression model interface
    """

    def __init__(self, model, feature_names=None, label_name=None,
                 norm_params=None, normalize=True, one_hot_categories=None):
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
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        """
        super().__init__(model, feature_names=feature_names,
                         label_names=label_name, norm_params=norm_params,
                         normalize=normalize,
                         one_hot_categories=one_hot_categories)

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
            prediction = PreProcess.unnormalize(prediction, means, stdevs)

        return prediction

    def _parse_labels(self, label, name=None):
        """
        Parse labels and normalize if desired

        Parameters
        ----------
        label : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        name : list, optional
            List of label names, by default None

        Returns
        -------
        label : ndarray
            Parsed labels array, normalized if desired
        """
        if self.normalize_labels:
            label = super()._parse_labels(label, names=name)

        if len(self.label_names) > 1:
            msg = ("Only a single label can be supplied to {}, but {} were"
                   .format(self.__class__.__name__, len(self.label_names)))
            logger.error(msg)
            raise ValueError(msg)

        return label

    def train_model(self, features, label, shuffle=True, parse_kwargs=None,
                    fit_kwargs=None):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Input features to train on
        label : dict | pandas.DataFrame
            label to train on
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        parse_kwargs : dict
            kwargs for cls._parse_features
        fit_kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor.fit
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        features = self._parse_features(features, **parse_kwargs)

        label = self._parse_labels(label)

        if fit_kwargs is None:
            fit_kwargs = {}

        if shuffle:
            L = len(features)
            i = np.random.choice(L, size=L, replace=False)
            features = features[i]
            label = label[i]

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
                        'label_name': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'normalize': (self.normalize_features,
                                      self.normalize_labels),
                        'one_hot_categories': self.one_hot_categories,
                        'model_params': self.model.get_params()}

        model_params = self.dict_json_convert(model_params)
        with open(path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

    @classmethod
    def build_trained(cls, features, label, normalize=True,
                      one_hot_categories=None, shuffle=True, save_path=None,
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
        normalize : bool | tuple, optional
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)
            by default True
        one_hot_categories : dict, optional
            Features to one-hot encode using given categories, if None do
            not run one-hot encoding, by default None
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
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

        model = cls.compile_model(**compile_kwargs)
        if one_hot_categories is not None:
            check_names = feature_names + label_name
            PreProcess.check_one_hot_categories(one_hot_categories,
                                                feature_names=check_names)
            feature_names = cls.make_one_hot_feature_names(feature_names,
                                                           one_hot_categories)

        model = cls(model, feature_names=feature_names, label_name=label_name,
                    normalize=normalize, one_hot_categories=one_hot_categories)

        model.train_model(features, label, shuffle=shuffle,
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
            Directory path to RandomForestModel from pickle file.

        Returns
        -------
        model : RandomForestModel
            Loaded RandomForestModel from disk.
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
        rf_params = model_params.pop('model_params')
        loaded = loaded.set_params(**rf_params)

        model = cls(loaded, **model_params)

        return model
