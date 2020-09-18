# -*- coding: utf-8 -*-
"""
Base Model Interface
"""
from abc import ABC
import logging
import numpy as np
import pandas as pd
from warnings import warn

from phygnn.utilities.pre_processing import PreProcess

logger = logging.getLogger(__name__)


class ModelBase(ABC):
    """
    Base Model Interface
    """

    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None, normalize=(True, False)):
        """
        Parameters
        ----------
        model : OBJ
            Sci-kit learn or tensorflow model
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
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
        """
        self._model = model

        if isinstance(feature_names, str):
            feature_names = [feature_names]
        elif isinstance(feature_names, np.ndarray):
            feature_names = feature_names.tolist()

        self._feature_names = feature_names

        if isinstance(label_names, str):
            label_names = [label_names]
        elif isinstance(label_names, np.ndarray):
            label_names = label_names.tolist()

        self._label_names = label_names
        if norm_params is None:
            norm_params = {}

        self._norm_params = norm_params
        self._normalize = self._parse_normalize(normalize)

    def __repr__(self):
        msg = "{}:\n{}".format(self.__class__.__name__, self.model_summary)

        return msg

    def __getitem__(self, features):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : pandas.DataFrame
            features to predict from

        Returns
        -------
        pandas.DataFrame
            label prediction
        """
        return self.predict(features)

    @property
    def model_summary(self):
        """
        Tensorflow model summary

        Returns
        -------
        str
        """
        try:
            summary = self._model.summary()
        except ValueError:
            summary = None

        return summary

    @property
    def normalize_features(self):
        """
        Flag to normalize features

        Returns
        -------
        bool
        """
        return self._normalize[0]

    @property
    def feature_names(self):
        """
        List of the feature variable names.

        Returns
        -------
        list
        """
        return self._feature_names

    @property
    def feature_dims(self):
        """
        Number of features

        Returns
        -------
        int
        """
        n_features = (len(self.feature_names)
                      if self.feature_names is not None else None)

        return n_features

    @property
    def normalize_labels(self):
        """
        Flag to normalize labels

        Returns
        -------
        bool
        """
        return self._normalize[1]

    @property
    def label_names(self):
        """
        label variable names

        Returns
        -------
        list
        """
        return self._label_names

    @property
    def label_dims(self):
        """
        Number of labels

        Returns
        -------
        int
        """
        n_labels = (len(self.label_names)
                    if self.label_names is not None else None)

        return n_labels

    @property
    def normalization_parameters(self):
        """
        Features and label (un)normalization parameters

        Returns
        -------
        dict
        """
        return self._norm_params

    @property
    def means(self):
        """
        Mapping feature/label names to the mean values for
        (un)normalization

        Returns
        -------
        dict
        """
        means = {k: v['mean'] for k, v in self._norm_params.items()}

        return means

    @property
    def stdevs(self):
        """
        Mapping feature/label names to the stdev values for
        (un)normalization

        Returns
        -------
        dict
        """
        stdevs = {k: v['stdev'] for k, v in self._norm_params.items()}

        return stdevs

    @property
    def model(self):
        """
        Trained model

        Returns
        -------
        tensorflow.keras.models
        """
        return self._model

    @property
    def feature_means(self):
        """
        Feature means, used for (un)normalization

        Returns
        -------
        list
        """
        means = None
        if self._feature_names is not None:
            means = []
            for f in self._feature_names:
                means.append(self.get_mean(f))

        return means

    @property
    def feature_stdevs(self):
        """
        Feature stdevs, used for (un)normalization

        Returns
        -------
        list
        """
        stdevs = None
        if self._feature_names is not None:
            stdevs = []
            for f in self._feature_names:
                stdevs.append(self.get_stdev(f))

        return stdevs

    @property
    def label_means(self):
        """
        label means, used for (un)normalization

        Returns
        -------
        list
        """
        means = None
        if self.label_names is not None:
            means = []
            for l_n in self.label_names:
                means.append(self.get_mean(l_n))

        return means

    @property
    def label_stdevs(self):
        """
        label stdevs, used for (un)normalization

        Returns
        -------
        list
        """
        stdevs = None
        if self.label_names is not None:
            stdevs = []
            for l_n in self.label_names:
                stdevs.append(self.get_stdev(l_n))

        return stdevs

    @staticmethod
    def _parse_normalize(normalize):
        """
        Parse normalize flag(s)

        Parameters
        ----------
        normalize : bool | tuple
            Boolean flag(s) as to whether features and labels should be
            normalized. Possible values:
            - True means normalize both
            - False means don't normalize either
            - Tuple of flags (normalize_feature, normalize_label)

        Returns
        -------
        normalize : tuple
            Boolean flags (normalize_feature, normalize_label)
        """
        if isinstance(normalize, bool):
            normalize = (normalize, normalize)
        elif isinstance(normalize, (tuple, list)):
            if len(normalize) != 2:
                msg = ('Expecting only 2 values: '
                       '(normalize_feature, normalize_label), but {} values '
                       'were provided!: {}'.format(len(normalize), normalize))
                logger.error(msg)
                raise ValueError(msg)
        else:
            msg = ('"normalize" must be a boolean flag or a tuple/list: '
                   '(normalize_feature, normalize_label), but {} was '
                   'provided!: {}'
                   .format(type(normalize), normalize))
            logger.error(msg)
            raise TypeError(msg)

        return tuple(normalize)

    @staticmethod
    def dict_json_convert(inp):
        """Recursively convert numeric values in dict to work with json dump

        Parameters
        ----------
        inp : dict
            Dictionary to convert.

        Returns
        -------
        out : dict
            Copy of dict input with all nested numeric values converted to
            base python int or float and all arrays converted to lists.
        """

        if isinstance(inp, dict):
            out = {k: ModelBase.dict_json_convert(v) for k, v in inp.items()}
        elif isinstance(inp, (list, tuple)):
            out = [ModelBase.dict_json_convert(i) for i in inp]
        elif np.issubdtype(type(inp), np.floating):
            out = float(inp)
        elif np.issubdtype(type(inp), np.integer):
            out = int(inp)
        elif isinstance(inp, np.ndarray):
            out = inp.tolist()
        else:
            out = inp

        return out

    @staticmethod
    def _parse_data(data, names=None):
        """
        Parse data array and names from input data

        Parameters
        ----------
        data : pandas.DataFrame | dict | ndarray
            Features/labels to parse
        names : list, optional
            List of data item names, by default None

        Returns
        -------
        data : ndarray
            Data array
        names: list
            List of data item names
        """
        if isinstance(data, pd.DataFrame):
            names = data.columns.values.tolist()
            data = data.values
        elif isinstance(data, dict):
            names = list(data.keys())
            data = np.dstack(list(data.values()))[0]
        elif isinstance(data, np.ndarray):
            if names is None:
                msg = ('Names of items must be supplied to parse data '
                       'arrays')
                logger.error(msg)
                raise RuntimeError(msg)

        return data, names

    @staticmethod
    def _get_item_number(arr):
        """
        Get number of items in array (labels or features)

        Parameters
        ----------
        arr : ndarray
            1 or 2D array

        Returns
        -------
        n : int
            Number of items
        """
        if len(arr.shape) == 1:
            n = 1
        else:
            n = arr.shape[1]

        return n

    def get_norm_params(self, names):
        """
        Get means and stdevs for given feature/label names

        Parameters
        ----------
        names : list
            list of feature/label names to get normalization params for

        Returns
        -------
        means : list
            List of means to use for (un)normalization
        stdevs : list
            List of stdevs to use for (un)normalization
        """
        means = []
        stdevs = []
        for name in names:
            means.append(self.get_mean(name))
            stdevs.append(self.get_stdev(name))

        if not all(set(means)):
            means = None

        if not all(set(stdevs)):
            stdevs = None

        return means, stdevs

    def get_mean(self, name):
        """
        Get feature | label mean

        Parameters
        ----------
        name : str
            feature | label name

        Returns
        -------
        mean : float
            Mean value used for normalization
        """
        mean = self._norm_params.get(name, None)
        if mean is not None:
            mean = mean.get('mean', None)

        return mean

    def get_stdev(self, name):
        """
        Get feature | label stdev

        Parameters
        ----------
        name : str
            feature | label name

        Returns
        -------
        stdev : float
            Stdev value used for normalization
        """
        stdev = self._norm_params.get(name, None)
        if stdev is not None:
            stdev = stdev.get('stdev', None)

        return stdev

    def _normalize_dict(self, items):
        """
        Normalize given dictionary of items (features | labels)

        Parameters
        ----------
        items : dict
            mapping of names to vectors

        Returns
        -------
        norm_items : dict
            mapping of names to normalized-feature vectors
        """
        norm_items = {}
        for key, value in items.items():
            mean = self.get_mean(key)
            stdev = self.get_stdev(key)
            update = mean is None or stdev is None
            try:
                value, mean, stdev = PreProcess.normalize(value, mean=mean,
                                                          stdev=stdev)
                if update:
                    norm_params = {key: {'mean': mean, 'stdev': stdev}}
                    self._norm_params.update(norm_params)
            except Exception as ex:
                msg = "Could not normalize {}:\n{}".format(key, ex)
                logger.warning(msg)
                warn(msg)

            norm_items[key] = value

        return norm_items

    def _normalize_df(self, df):
        """
        Normalize DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of features/label to normalize

        Returns
        -------
        norm_df : pandas.DataFrame
            Normalized features/label
        """
        means, stdevs = self.get_norm_params(df.columns)
        update = means is None or stdevs is None

        norm_df, means, stdevs = PreProcess.normalize(df, mean=means,
                                                      stdev=stdevs)
        if update:
            for i, c in enumerate(df.columns):
                norm_params = {c: {'mean': means[i], 'stdev': stdevs[i]}}
                self._norm_params.update(norm_params)

        return norm_df

    def _normalize_arr(self, arr, names):
        """
        Normalize array and save normalization parameters to given names

        Parameters
        ----------
        arr : ndarray
            Array of features/label to normalize
        names : list
            List of feature/label names

        Returns
        -------
        norm_arr : ndarray
            Normalized features/label
        """
        n_names = self._get_item_number(arr)
        if len(names) != n_names:
            msg = ("Number of item names ({}) does not match number of items "
                   "({})".format(len(names), arr.shape[1]))
            logger.error(msg)
            raise RuntimeError(msg)

        means, stdevs = self.get_norm_params(names)
        update = means is None or stdevs is None

        norm_arr, means, stdevs = PreProcess.normalize(arr, mean=means,
                                                       stdev=stdevs)
        if update:
            for i, n in enumerate(names):
                norm_params = {n: {'mean': means[i], 'stdev': stdevs[i]}}
                self._norm_params.update(norm_params)

        return norm_arr

    def normalize(self, data, names=None):
        """
        Normalize given data

        Parameters
        ----------
        data : dict | pandas.DataFrame | ndarray
            Data to normalize
        names : list, optional
            List of data item names, needed to normalized ndarrays,
            by default None

        Returns
        -------
        data : dict | pandas.DataFrame | ndarray
            Normalized data in same format as input
        """
        if isinstance(data, dict):
            data = self._normalize_dict(data)
        elif isinstance(data, pd.DataFrame):
            data = self._normalize_df(data)
        elif isinstance(data, (list, np.ndarray)):
            if names is None:
                msg = ('Names of items must be supplied to nomralize data '
                       'arrays')
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                data = self._normalize_arr(data, names)
        else:
            msg = "Cannot normalize data of type: {}".format(type(data))
            logger.error(msg)
            raise RuntimeError(msg)

        return data

    def _unnormalize_dict(self, items):
        """
        Un-normalize given dictionary of items (features | labels)

        Parameters
        ----------
        items : dict
            mapping of names to vectors

        Returns
        -------
        native_items : dict
            mapping of names to native vectors
        """
        native_items = {}
        for key, value in items.items():
            norm_params = self.normalization_parameters[key]
            if norm_params is not None:
                value = PreProcess.unnormalize(value, norm_params['mean'],
                                               norm_params['stdev'])
            else:
                msg = ("Normalization Parameters unavailable, {} will not be "
                       "un-normalized!".format(key))
                logger.warning(msg)
                warn(msg)

            native_items[key] = value

        return native_items

    def _unnormalize_df(self, df):
        """
        Un-normalize DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame of features/label to un-normalize

        Returns
        -------
        df : pandas.DataFrame
            Native features/label df if norm params are not None
        """
        means, stdevs = self.get_norm_params(df.columns)

        if means is not None and stdevs is not None:
            df = PreProcess.unnormalize(df.copy(), means, stdevs)
        else:
            msg = ("Normalization parameters are unavailable, df will not be "
                   "un-normalized!")
            logger.warning(msg)
            warn(msg)

        return df

    def _unnormalize_arr(self, arr, names):
        """
        Un-normalize array using given names

        Parameters
        ----------
        arr : ndarray
            Array of features/label to un-normalize
        names : list
            List of feature/label names

        Returns
        -------
        arr : ndarray
            Native features/label array if norm params are not None
        """
        n_names = self._get_item_number(arr)
        if len(names) != n_names:
            msg = ("Number of item names ({}) does not match number of items "
                   "({})".format(len(names), arr.shape[1]))
            logger.error(msg)
            raise RuntimeError(msg)

        means, stdevs = self.get_norm_params(names)

        if means is not None and stdevs is not None:
            arr = PreProcess.unnormalize(arr.copy(), means, stdevs)
        else:
            msg = ("Normalization parameters are unavailable, arr will not be "
                   "un-normalized!")
            logger.warning(msg)
            warn(msg)

        return arr

    def unnormalize(self, data, names=None):
        """
        Un-normalize given data

        Parameters
        ----------
        data : dict | pandas.DataFrame | ndarray
            Data to un-normalize
        names : list, optional
            List of data item names, needed to un-normalized ndarrays,
            by default None

        Returns
        -------
        data : dict | pandas.DataFrame | ndarray
            Native data in same format as input
        """
        if isinstance(data, dict):
            data = self._unnormalize_dict(data)
        elif isinstance(data, pd.DataFrame):
            data = self._unnormalize_df(data)
        elif isinstance(data, (list, np.ndarray)):
            if names is None:
                msg = ('Names of items must be supplied to un-nomralize data '
                       'arrays')
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                data = self._unnormalize_arr(data, names)
        else:
            msg = "Cannot un-normalize data of type: {}".format(type(data))
            logger.error(msg)
            raise RuntimeError(msg)

        return data

    def _check_one_hot_norm_params(self, one_hot_features):
        """
        Check one hot feature normalization parameters to ensure they are
        {mean: 0, stdev: 1} to prevent normalization

        Parameters
        ----------
        one_hot_features : list
            list of one hot features
        """
        for feature in one_hot_features:
            mean = self.get_mean(feature)
            stdev = self.get_stdev(feature)
            if mean != 0 and stdev != 1:
                norm_params = {feature: {'mean': 0, 'stdev': 1}}
                self._norm_params.update(norm_params)

    def _parse_features(self, features, names=None, process_one_hot=True,
                        **kwargs):
        """
        Parse features

        Parameters
        ----------
        features : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        names : list, optional
            List of feature names, by default None
        process_one_hot : bool, optional
            Check for and process one-hot variables, by default True
        kwargs : dict, optional
            kwargs for PreProcess.one_hot

        Returns
        -------
        features : ndarray
            Parsed features array normalized and with str columns converted
            to one hot vectors if desired
        """
        features, feature_names = self._parse_data(features, names=names)

        if len(features.shape) != 2:
            msg = ('{} can only use 2D data as input!'
                   .format(self.__class__.__name__))
            logger.error(msg)
            raise RuntimeError(msg)

        if self.feature_names is not None:
            if features.shape[1] != len(self.feature_names):
                msg = ('data has {} features but expected {}'
                       .format(features.shape[1], self.feature_dims))
                logger.error(msg)
                raise RuntimeError(msg)

        if self._feature_names is None:
            self._feature_names = feature_names
        elif self.feature_names != feature_names:
            msg = ('Expecting features with names: {}, but was provided with: '
                   '{}!'.format(feature_names, self.feature_names))
            logger.error(msg)
            raise RuntimeError(msg)

        if process_one_hot:
            kwargs.update({'return_ind': True})
            features, one_hot_ind = PreProcess.one_hot(features, **kwargs)
            if one_hot_ind:
                one_hot_features = [self.feature_names[i] for i in one_hot_ind]
                self._check_one_hot_norm_params(one_hot_features)

        if self.normalize_features:
            features = self.normalize(features, names=feature_names)

        return features

    def _parse_labels(self, labels, names=None):
        """
        Parse labels and normalize if desired

        Parameters
        ----------
        labels : pandas.DataFrame | dict | ndarray
            Features to train on or predict from
        names : list, optional
            List of label names, by default None

        Returns
        -------
        labels : ndarray
            Parsed labels array, normalized if desired
        """
        labels, label_names = self._parse_data(labels, names=names)

        if self.label_names is not None:
            n_labels = self._get_item_number(labels)
            if n_labels != len(self.label_names):
                msg = ('data has {} labels but expected {}'
                       .format(labels.shape[1], self.label_dims))
                logger.error(msg)
                raise RuntimeError(msg)

        if self._label_names is None:
            self._label_names = label_names
        elif self.label_names != label_names:
            msg = ('Expecting labels with names: {}, but was provided with: '
                   '{}!'.format(label_names, self.label_names))
            logger.error(msg)
            raise RuntimeError(msg)

        if self.normalize_labels:
            labels = self.normalize(labels, names=label_names)

        return labels

    def predict(self, features, table=True, parse_kwargs=None,
                predict_kwargs=None):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : dict | pandas.DataFrame
            features to predict from
        table : bool, optional
            Return pandas DataFrame
        parse_kwargs : dict
            kwargs for cls._parse_features
        predict_wargs : dict
            kwargs for tensorflow.*.predict

        Returns
        -------
        prediction : ndarray | pandas.DataFrame
            label prediction
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        if isinstance(features, np.ndarray):
            parse_kwargs.update({"names": self.feature_names})

        features = self._parse_features(features, **parse_kwargs)

        if predict_kwargs is None:
            predict_kwargs = {}

        prediction = self._model.predict(features, **predict_kwargs)
        if self.normalize_labels:
            prediction = self.unnormalize(prediction, names=self.label_names)

        if table:
            prediction = pd.DataFrame(prediction, columns=self.label_names)

        return prediction
