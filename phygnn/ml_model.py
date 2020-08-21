# -*- coding: utf-8 -*-
"""
TensorFlow Model
"""
import json
import logging
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from warnings import warn

logger = logging.getLogger(__name__)


class MLModelBase:
    """
    Machine Learning Model Base
    """

    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None):
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
        """
        self._model = model

        if isinstance(feature_names, str):
            feature_names = [feature_names]

        self._feature_names = feature_names

        if isinstance(label_names, str):
            label_names = [label_names]

        self._label_names = label_names
        if norm_params is None:
            norm_params = {}

        self._norm_params = norm_params

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
    def feature_names(self):
        """
        List of the feature variable names.

        Returns
        -------
        list
        """
        return self._feature_names

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
        dict
        """
        means = None
        if self._feature_names is not None:
            means = {}
            for f in self._feature_names:
                v = self._norm_params.get(f, None)
                if v is not None:
                    means[f] = v['mean']

        return means

    @property
    def feature_stdevs(self):
        """
        Feature stdevs, used for (un)normalization

        Returns
        -------
        dict
        """
        stdevs = None
        if self._feature_names is not None:
            stdevs = {}
            for f in self._feature_names:
                v = self._norm_params.get(f, None)
                if v is not None:
                    stdevs[f] = v['stdev']

        return stdevs

    @property
    def label_means(self):
        """
        label means, used for (un)normalization

        Returns
        -------
        dict
        """
        means = None
        if self.label_names is not None:
            means = {}
            for l_n in self.label_names:
                v = self._norm_params.get(l_n, None)
                if v is not None:
                    means[l_n] = v['mean']

        return means

    @property
    def label_stdevs(self):
        """
        label stdevs, used for (un)normalization

        Returns
        -------
        dict
        """
        stdevs = None
        if self.label_names is not None:
            stdevs = {}
            for l_n in self.label_names:
                v = self._norm_params.get(l_n, None)
                if v is not None:
                    stdevs[l_n] = v['stdev']

        return stdevs

    @staticmethod
    def _normalize(native_arr, mean=None, stdev=None):
        """
        Normalize features with mean at 0 and stdev of 1.

        Parameters
        ----------
        native_arr : ndarray
            native data
        mean : float | None
            mean to use for normalization
        stdev : float | None
            stdev to use for normalization

        Returns
        -------
        norm_arr : ndarray
            normalized data
        mean : float
            mean used for normalization
        stdev : float
            stdev used for normalization
        """

        if mean is None:
            mean = np.nanmean(native_arr, axis=0)

        if stdev is None:
            stdev = np.nanstd(native_arr, axis=0)

        norm_arr = native_arr - mean
        norm_arr /= stdev

        return norm_arr, mean, stdev

    @staticmethod
    def _unnormalize(norm_arr, mean, stdev):
        """
        Unnormalize data with mean at 0 and stdev of 1.

        Parameters
        ----------
        norm_arr : ndarray
            normalized data
        mean : float
            mean used for normalization
        stdev : float
            stdev used for normalization

        Returns
        -------
        native_arr : ndarray
            native un-normalized data
        """
        native_arr = norm_arr * stdev
        native_arr += mean

        return native_arr

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
            out = {k: MLModelBase.dict_json_convert(v) for k, v in inp.items()}
        elif isinstance(inp, (list, tuple)):
            out = [MLModelBase.dict_json_convert(i) for i in inp]
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
    def _parse_data(data):
        """
        Parse features or labels

        Parameters
        ----------
        data : dict | pandas.DataFrame
            Features or label to use with model

        Returns
        -------
        data : dict
            Dictionary of normalized (if desired) features or label
        """

        return data

    def get_norm_params(self, name):
        """
        Feature or label normalization parameters

        Parameters
        ----------
        name : str
            feature | label name

        Returns
        -------
        dict
            mean and stdev values for given feature | label
        """
        return self._norm_params.get(name, None)

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

    def normalize(self, items):
        """
        Normalize given items (features | labels)

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
            try:
                value, mean, stdev = self._normalize(value, mean=mean,
                                                     stdev=stdev)
                norm_params = {key: {'mean': mean, 'stdev': stdev}}
                self._norm_params.update(norm_params)
            except Exception as ex:
                msg = "Could not normalize {}:\n{}".format(key, ex)
                logger.warning(msg)
                warn(msg)

            norm_items[key] = value

        return norm_items

    def unnormalize(self, items):
        """
        Unnormalize given items (features | labels)

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
            norm_params = self.get_norm_params(key)
            if norm_params is not None:
                value = self._unnormalize(value, norm_params['mean'],
                                          norm_params['stdev'])
                native_items[key] = value
            else:
                msg = ("Normalization Parameters unavailable for {}"
                       .format(key))
                logger.warning(msg)
                warn(msg)

        return native_items

    def unnormalize_prediction(self, prediction):
        """
        Unnormalize prediction if needed

        Parameters
        ----------
        prediction : ndarray
            TfModel prediction

        Returns
        -------
        prediction : ndarray
            Unnormalized prediction
        """
        out = np.empty(prediction.shape, prediction.dtype)
        for ax, label in enumerate(self.label_names):
            value = prediction[:, ax]
            norm_params = self.get_norm_params(label)
            if norm_params is not None:
                value = self._unnormalize(value, norm_params['mean'],
                                          norm_params['stdev'])
            else:
                msg = ("Normalization Parameters unavailable for {}"
                       .format(label))
                logger.warning(msg)
                warn(msg)

            out[:, ax] = value

        return out

    def predict(self, features, **kwargs):
        """
        Use model to predict label from given features

        Parameters
        ----------
        features : dict | pandas.DataFrame
            features to predict from
        kwargs : dict
            kwargs for tensorflow.*.predict

        Returns
        -------
        prediction : dict
            label prediction
        """
        features = self._parse_data(features)

        prediction = pd.DataFrame(self._model.predict(features, **kwargs),
                                  columns=self.label_names)

        prediction = self.unnormalize_prediction(prediction)

        return prediction


class TfModel(MLModelBase):
    """
    TensorFlow Keras Model
    """
    def __init__(self, model, feature_names=None, label_names=None,
                 norm_params=None):
        """
        Parameters
        ----------
        model : tensorflow.keras.models.Sequential
            Tensorflow Keras Model
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        norm_params : dict, optional
            Dictionary mapping feature and label names (keys) to normalization
            parameters (mean, stdev), by default None
        """
        super().__init__(model, feature_names=feature_names,
                         label_names=label_names, norm_params=norm_params)

        self._history = None

    @property
    def history(self):
        """
        Model training history

        Returns
        -------
        pandas.DataFrame | None
        """
        if self._history is None:
            msg = 'Model has not been trained yet!'
            logger.warning(msg)
            warn(msg)
            history = None
        else:
            history = pd.DataFrame(self._history.history)
            history['epoch'] = self._history.epoch

        return history

    @staticmethod
    def _clean_name(name):
        """
        Make feature | label name compatible with TensorFlow

        Parameters
        ----------
        name : str
            Feature |label name from GOOML

        Returns
        -------
        name : str
            Feature | label name compatible with TensorFlow
        """
        name = name.replace(' ', '_')
        name = name.replace('*', '-x-')
        name = name.replace('+', '-plus-')
        name = name.replace('**', '-exp-')
        name = name.replace(')', '')
        name = name.replace('log(', 'log-')

        return name

    @staticmethod
    def _generate_feature_columns(features):
        """
        Generate feature layer from features table

        Parameters
        ----------
        features : dict
            model features

        Returns
        -------
        feature_columns : list
            List of tensorFlow.feature_column objects
        """
        feature_columns = []
        for name, data in features.items():
            name = TfModel._clean_name(name)
            if np.issubdtype(data.dtype.name, np.number):
                f_col = feature_column.numeric_column(name)
            else:
                f_col = TfModel._generate_cat_column(name, data)

            feature_columns.append(f_col)

        return feature_columns

    @staticmethod
    def _generate_cat_column(name, data, vocab_threshold=50, bucket_size=100):
        """Generate a feature column from a categorical string data set

        Parameters
        ----------
        name : str
            Name of categorical columns
        data : np.ndarray | list
            String data array
        vocab_threshold : int
            Number of unique entries in the data array below which this
            will use a vocabulary list, above which a hash bucket will be used.
        bucket_size : int
            Hash bucket size.

        Returns
        -------
        f_col : IndicatorColumn
            Categorical feature column.
        """

        n_unique = len(set(data))

        if n_unique < vocab_threshold:
            f_col = feature_column.categorical_column_with_vocabulary_list(
                name, list(set(data)))
        else:
            f_col = feature_column.categorical_column_with_hash_bucket(
                name, bucket_size)

        f_col = feature_column.indicator_column(f_col)

        return f_col

    @staticmethod
    def _build_feature_columns(feature_columns):
        """
        Build the feature layer from given feature column descriptions

        Parameters
        ----------
        feature_columns : list
            list of feature column descriptions (dictionaries)

        Returns
        -------
        tf_columns : list
            List of tensorFlow.feature_column objects
        """
        tf_columns = {}
        col_map = {}  # TODO: build map to tf.feature_column functions
        # TODO: what feature_columns need to be wrapped
        indicators = [feature_column.categorical_column_with_hash_bucket,
                      feature_column.categorical_column_with_identity,
                      feature_column.categorical_column_with_vocabulary_file,
                      feature_column.categorical_column_with_vocabulary_list,
                      feature_column.crossed_column]
        for col in feature_columns:
            name = col['name']
            f_type = col_map.get(col['type'], col['type'])
            kwargs = col.get('kwargs', {})

            if f_type == feature_column.crossed_column:
                cross_cols = [tf_columns[name]
                              for name in col['cross_columns']]
                f_col = f_type(cross_cols, **kwargs)
            elif f_type == feature_column.embedding_column:
                embedded_type = col_map[col['embedded_col']]
                f_col = embedded_type(name, **kwargs)
                f_col = f_type(f_col, **kwargs)
            else:
                f_col = f_type(name, **kwargs)

            if f_type in indicators:
                f_col = feature_column.indicator_column(f_col)

            tf_columns[name] = f_col

        return tf_columns

    @staticmethod
    def _compile_model(feature_columns, model_layers=None, learning_rate=0.001,
                       loss="mean_squared_error", metrics=('mae', 'mse'),
                       optimizer_class=None, **kwargs):
        """
        Build tensorflow sequential model from given layers and kwargs

        Parameters
        ----------
        feature_columns : list
            List of tensorFlow.feature_column objects
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : None | tf.keras.optimizers
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel._compile_model() method
            The default is the RMSprop optimizer.
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        tensorflow.keras.models.Sequential
            Compiled tensorflow Sequential model
        """
        model = tf.keras.models.Sequential()
        model.add(layers.DenseFeatures(feature_columns))
        if model_layers is None:
            # Add a single linear layer
            model.add(layers.Dense(units=1, input_shape=(1,)))
        else:
            for layer in model_layers:
                dropout = layer.pop('dropout', None)
                model.add(layers.Dense(**layer))

                if dropout is not None:
                    model.add(layers.Dropout(dropout))

        if isinstance(metrics, tuple):
            metrics = list(metrics)
        elif not isinstance(metrics, list):
            metrics = [metrics]

        if optimizer_class is None:
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate)
        else:
            optimizer = optimizer_class(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      **kwargs)

        return model

    @staticmethod
    def build_model(features, feature_columns=None, model_layers=None,
                    learning_rate=0.001, loss="mean_squared_error",
                    metrics=('mae', 'mse'), optimizer_class=None, **kwargs):
        """
        Build tensorflow sequential model from given layers and kwargs

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Model features
        feature_columns : list, optional
            list of feature column descriptions (dictionaries)
            if None use numeric columns with the feature_attr name,
            by default None
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : None | tf.keras.optimizers
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel._compile_model() method.
            The default is the RMSprop optimizer.
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        tensorflow.keras.models.Sequential
            Compiled tensorflow Sequential model
        """
        if feature_columns is None:
            feature_columns = TfModel._generate_feature_columns(features)
        else:
            if len(feature_columns) < len(features.keys()):
                msg = ("There must be at least one feature column per feature!"
                       " {} columns were supplied but there are {} features!"
                       .format(len(feature_columns),
                               len(features.keys())))
                logger.error(msg)
                raise ValueError

            feature_columns = TfModel._build_feature_columns(feature_columns)

        model = TfModel._compile_model(feature_columns,
                                       model_layers=model_layers,
                                       learning_rate=learning_rate,
                                       loss=loss, metrics=metrics,
                                       optimizer_class=optimizer_class,
                                       **kwargs)

        return model

    def _parse_data(self, data, normalize=True, clean_names=True):
        """
        Parse features or labels, normalize, and clean names if requested

        Parameters
        ----------
        data : dict | pandas.DataFrame
            Features or label to use with model
        normalize : bool, optional
            Flag to normalize features or labels, by default True
        clean_names : bool, optional
            Flag to clean feature or label names, by default True

        Returns
        -------
        data : dict
            Dictionary of normalized (if desired) features or label
        """
        if isinstance(data, pd.DataFrame):
            data = {name: np.array(value) for name, value in data.items()}
        elif not isinstance(data, dict):
            msg = ("Features and label must be supplied as a pandas.DataFrame"
                   " or python dictionary, but recieved: {}"
                   .format(type(data)))
            logger.error(msg)
            raise ValueError(msg)

        if normalize:
            data = self.normalize(data)

        if clean_names:
            data = {self._clean_name(key): value
                    for key, value in data.items()}

        return data

    def train_model(self, features, labels, norm_label=True, epochs=100,
                    validation_split=0.2, early_stop=True, **kwargs):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Input features to train on
        labels : dict | pandas.DataFrame
            label to train on
        norm_label : bool
            Flag to normalize label
        epochs : int, optional
            Number of epochs to train the model, by default 100
        validation_split : float, optional
            Fraction of the training data to be used as validation data,
            by default 0.2
        early_stop : bool
            Flag to stop training when it stops improving
        kwargs : dict
            kwargs for tensorflow.keras.models.fit
        """
        features = self._parse_data(features)
        self._feature_names = list(features.keys())

        labels = self._parse_data(labels, normalize=norm_label)

        self._label_names = list(labels.keys())

        if self._history is not None:
            msg = 'Model has already been trained and will be re-fit!'
            logger.warning(msg)
            warn(msg)

        if early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=10)
            callbacks = kwargs.pop('callbacks', None)
            if callbacks is None:
                callbacks = [early_stop]
            else:
                callbacks.append(early_stop)

            kwargs['callbacks'] = callbacks

        if validation_split > 0:
            split = int(len(list(features.values())[0]) * validation_split)
            validate_features = {name: arr[-split:]
                                 for name, arr in features.items()}
            validate_labels = [arr[-split:] for arr in labels.values()]
            validation_data = (validate_features, validate_labels)

            features = {name: arr[:-split]
                        for name, arr in features.items()}
            labels = [arr[:-split] for arr in labels.values()]
        else:
            validation_data = None

        self._history = self._model.fit(x=features, y=labels, epochs=epochs,
                                        validation_data=validation_data,
                                        **kwargs)

    def save_model(self, path):
        """
        Save TfModel to path.

        Parameters
        ----------
        path : str
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json.
        """
        if path.endswith('.json'):
            path = path.replace('.json', '/')

        if not path.endswith('/'):
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        tf.saved_model.save(self.model, path)

        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters}

        json_path = path.rstrip('/') + '.json'
        model_params = self.dict_json_convert(model_params)
        with open(json_path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

    @classmethod
    def build(cls, features, feature_columns=None, model_layers=None,
              learning_rate=0.001, loss="mean_squared_error",
              metrics=('mae', 'mse'), optimizer_class=None, **kwargs):
        """
        Build tensorflow sequential model from given features, layers and
        kwargs

        Parameters
        ----------
         features : dict | pandas.DataFrame
            Model features
        feature_columns : list, optional
            list of feature column descriptions (dictionaries)
            if None use numeric columns with the feature_attr name,
            by default None
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : None | tf.keras.optimizers
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel._compile_model() method
            The default is the RMSprop optimizer.
        kwargs : dict
            kwargs for tensorflow.keras.models.compile

        Returns
        -------
        model : TfModel
            Initialized TfKeraModel obj
        """
        model = TfModel.build_model(features, feature_columns=feature_columns,
                                    model_layers=model_layers,
                                    learning_rate=learning_rate, loss=loss,
                                    metrics=metrics,
                                    optimizer_class=optimizer_class, **kwargs)

        return cls(model)

    @classmethod
    def train(cls, features, labels, feature_columns=None, model_layers=None,
              learning_rate=0.001, loss="mean_squared_error",
              metrics=('mae', 'mse'), optimizer_class=None, norm_label=True,
              epochs=100, validation_split=0.2, early_stop=True,
              save_path=None, build_kwargs=None, train_kwargs=None):
        """
        Build tensorflow sequential model from given features, layers and
        kwargs and then train with given label and kwargs

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Model features
        labels : dict | pandas.DataFrame
            label to train on
        feature_columns : list, optional
            list of feature column descriptions (dictionaries)
            if None use numeric columns with the feature_attr name,
            by default None
        model_layers : list, optional
            List of tensorflow layers.Dense kwargs (dictionaries)
            if None use a single linear layer, by default None
        learning_rate : float, optional
            tensorflow optimizer learning rate, by default 0.001
        loss : str, optional
            name of objective function, by default "mean_squared_error"
        metrics : list, optional
            List of metrics to be evaluated by the model during training and
            testing, by default ('mae', 'mse')
        optimizer_class : None | tf.keras.optimizers
            Optional explicit request of optimizer. This should be a class
            that will be instantated in the TfModel._compile_model() method
            The default is the RMSprop optimizer.
        norm_label : bool
            Flag to normalize label
        epochs : int, optional
            Number of epochs to train the model, by default 100
        validation_split : float, optional
            Fraction of the training data to be used as validation data,
            by default 0.2
        early_stop : bool
            Flag to stop training when it stops improving
        save_path : str
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json.
        build_kwargs : dict
            kwargs for tensorflow.keras.models.compile
        train_kwargs : dict
            kwargs for tensorflow.keras.models.fit

        Returns
        -------
        model : TfModel
            Initialized and trained TfModel obj
        """
        if build_kwargs is None:
            build_kwargs = {}

        model = cls.build(features, feature_columns=feature_columns,
                          model_layers=model_layers,
                          learning_rate=learning_rate, loss=loss,
                          metrics=metrics, optimizer_class=optimizer_class,
                          **build_kwargs)

        if train_kwargs is None:
            train_kwargs = {}

        model.train_model(features, labels, norm_label=norm_label,
                          epochs=epochs, validation_split=validation_split,
                          early_stop=early_stop, **train_kwargs)

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
        if path.endswith('.json'):
            path = path.replace('.json', '/')

        if not path.endswith('/'):
            path += '/'

        if not os.path.isdir(path):
            e = ('Can only load directory path but target is not '
                 'directory: {}'.format(path))
            logger.error(e)
            raise IOError(e)

        loaded = tf.keras.models.load_model(path)

        json_path = path.rstrip('/') + '.json'
        with open(json_path, 'r') as f:
            model_params = json.load(f)

        model = cls(loaded, **model_params)

        return model


class RandomForestModel(MLModelBase):
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
                         label_name=label_name, norm_params=norm_params)

        if len(self.label_names) > 1:
            msg = ("Only a single label can be supplied to {}, but {} were"
                   .format(self.__class__.__name__, len(self.label_names)))
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def build_model(**kwargs):
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

    def _get_norm_params(self, names):
        """
        Get means and stdevs for given feature/label names

        Parameters
        ----------
        names : list
            list of feature/label names to get normalization params for

        Returns
        -------
        means : list | None
            List of means to use for (un)normalization
        stdevs : list | None
            List of stdevs to use for (un)normalization
        """
        means = []
        stdevs = []
        for name in names:
            v = self._norm_params.get(name, None)
            if v is None:
                means = None
                stdevs = None
                break

            means.append(v['mean'])
            stdevs.append(v['stdev'])

        return means, stdevs

    def normalize(self, df):
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
        df = pd.get_dummies(df)
        means, stdevs = self._get_norm_params(df.columns)

        norm_df, means, stdevs = self._normalize(df, mean=means,
                                                 stdev=stdevs)
        for i, c in enumerate(df.columns):
            norm_params = {c: {'mean': means[i], 'stdev': stdevs[i]}}
            self._norm_params.update(norm_params)

        return norm_df

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

    def _parse_data(self, features, normalize=True, names=False):
        """
        Parse features or labels, normalize, and clean names if requested

        Parameters
        ----------
        features : panda.DataFrame
            Features or label to use with model
        normalize : bool, optional
            Flag to normalize features or labels, by default True
        names : bool, optional
            Flag to retain DataFrame, by default False

        Returns
        -------
        features : ndarray | panda.DataFrame
            Normalized (if desired) features or label
        """
        if not isinstance(features, pd.DataFrame):
            msg = ("Features must be a pandas.DataFrame, but {} was supplied"
                   .format(type(features)))
            logger.error(msg)
            raise ValueError(msg)

        if normalize:
            features = self.normalize(features)
        else:
            features = pd.get_dummies(features)

        if not names:
            features = features.values

        return features

    def train_model(self, features, label, norm_label=True, **kwargs):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : dict | pandas.DataFrame
            Input features to train on
        labels : dict | pandas.DataFrame
            label to train on
        norm_label : bool
            Flag to normalize label
        kwargs : dict
            kwargs for sklearn.ensemble.RandomForestRegressor.fit
        """
        features = self._parse_data(features, names=True)
        self._feature_names = list(features.columns)
        features = features.values

        label = self._parse_data(label, normalize=norm_label,
                                 names=True)
        self._label_names = list(label.columns)
        label = label.values

        if len(self.label_names) > 1:
            msg = ("Only a single label can be supplied to {}, but {} were"
                   .format(self.__class__.__name__, len(self.label_names)))
            logger.error(msg)
            raise ValueError(msg)

        self._model.fit(features, label.ravel(), **kwargs)

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
    def train(cls, features, labels, norm_label=True, save_path=None,
              build_kwargs=None, train_kwargs=None):
        """
        Build Random Forest Model with given kwargs and then train with
        given features, labels, and kwargs

        Parameters
        ----------
        features : pandas.DataFrame
            Model features
        labels : pandas.DataFrame
            label to train on
        norm_label : bool
            Flag to normalize label
        save_path : str
            Directory path to save model to. The RandomForest Model will be
            saved to the directory while the framework parameters will be
            saved in json.
        build_kwargs : dict
            kwargs for tensorflow.keras.models.compile
        train_kwargs : dict
            kwargs for tensorflow.keras.models.fit

        Returns
        -------
        model : RandomForestModel
            Initialized and trained RandomForestModel obj
        """
        if build_kwargs is None:
            build_kwargs = {}

        model = cls(cls.build_model(**build_kwargs))

        if train_kwargs is None:
            train_kwargs = {}

        model.train_model(features, labels, norm_label=norm_label,
                          **train_kwargs)

        if save_path is not None:
            pass
            # model.save_model(save_path)

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
