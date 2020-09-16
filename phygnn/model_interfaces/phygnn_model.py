# -*- coding: utf-8 -*-
"""
TensorFlow Model
"""
import json
import logging
import os

from phygnn.phygnn import PhysicsGuidedNeuralNetwork
from phygnn.model_interfaces.base_model import ModelBase

logger = logging.getLogger(__name__)


class PhygnnModel(ModelBase):
    """
    Phygnn Model interface
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

    @property
    def layers(self):
        """
        Model layers

        Returns
        -------
        list
        """
        return self.model.layers

    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        return self.model.weights

    @property
    def kernel_weights(self):
        """
        Get a list of the NN kernel weights (tensors)

        (can be used for kernel regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self.model.kernel_weights

    @property
    def bias_weights(self):
        """
        Get a list of the NN bias weights (tensors)

        (can be used for bias regularization).

        Does not include input layer or dropout layers.
        Does include the output layer.

        Returns
        -------
        list
        """
        return self.bias_weights

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self.model.history

    @staticmethod
    def compile_model(p_fun, loss_weights=(0.5, 0.5),
                      n_features=1, n_labels=1, hidden_layers=None,
                      metric='mae', initializer=None, optimizer=None,
                      learning_rate=0.01, history=None,
                      kernel_reg_rate=0.0, kernel_reg_power=1,
                      bias_reg_rate=0.0, bias_reg_power=1,
                      feature_names=None, output_names=None):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This function must take (y_predicted, y_true, p, **p_kwargs)
            as arguments with datatypes (tf.Tensor, np.ndarray, np.ndarray).
            The function must return a tf.Tensor object with a single numeric
            loss value (output.ndim == 0).
        loss_weights : tuple, optional
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        n_features : int, optional
            Number of input features.
        n_labels : int, optional
            Number of output labels.
        hidden_layers : list, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers (9 layers total):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        initializer : tensorflow.keras.initializers, optional
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers, optional
            Instantiated neural network optimization object.
            None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).
        feature_names : list | tuple | None, optional
            Training feature names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared feature names, making it
            easier to feed in features for prediction. This will also get set
            if phygnn is trained on a DataFrame.
        output_names : list | tuple | None, optional
            Prediction output names (strings). Mostly a convenience so that a
            loaded-from-disk model will have declared output names, making it
            easier to understand prediction output. This will also get set
            if phygnn is trained on a DataFrame.

        Returns
        -------
        model : PhysicsGuidedNeuralNetwork
            Instantiated phygnn model
        """
        model = PhysicsGuidedNeuralNetwork(p_fun,
                                           loss_weights=loss_weights,
                                           n_features=n_features,
                                           n_labels=n_labels,
                                           hidden_layers=hidden_layers,
                                           metric=metric,
                                           initializer=initializer,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           history=history,
                                           kernel_reg_rate=kernel_reg_rate,
                                           kernel_reg_power=kernel_reg_power,
                                           bias_reg_rate=bias_reg_rate,
                                           bias_reg_power=bias_reg_power,
                                           feature_names=feature_names,
                                           output_names=output_names)

        return model

    def train_model(self, features, labels, p, n_batch=16, n_epoch=10,
                    shuffle=True, validation_split=0.2, run_preflight=True,
                    return_diagnostics=False, p_kwargs=None, parse_kwargs=None,
                    norm_labels=True):
        """
        Train the model with the provided features and label

        Parameters
        ----------
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame. If this is a
            DataFrame, the index is ignored, the columns are used with
            self.output_names, and the df is converted into a numpy array for
            batching and passing to the training algorithm.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. If this is a DataFrame, the index and column labels
            are ignored and the df is converted into a numpy array for batching
            and passing to the training algorithm and physical loss function.
        n_batch : int
            Number of times to update the NN weights per epoch (number of
            mini-batches). The training data will be split into this many
            mini-batches and the NN will train on each mini-batch, update
            weights, then move onto the next mini-batch.
        n_epoch : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        validation_split : float
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        parse_kwargs : dict
            kwargs for cls._parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True

        Returns
        -------
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """
        if parse_kwargs is None:
            parse_kwargs = {}

        x = self._parse_features(features, **parse_kwargs)
        y = self._parse_labels(labels, normalize=norm_labels)

        diagnostics = self.model.fit(x, y, p,
                                     n_batch=n_batch,
                                     n_epoch=n_epoch,
                                     shuffle=shuffle,
                                     validation_split=validation_split,
                                     p_kwargs=p_kwargs,
                                     run_preflight=run_preflight,
                                     return_diagnostics=return_diagnostics)

        return diagnostics

    def save_model(self, path):
        """
        Save phygnn model to path.

        Parameters
        ----------
        path : str
            Save phygnn model to pickle file.
        """
        if path.endswith('.json'):
            path = path.replace('.json', '/')

        if not path.endswith('/'):
            path += '/'

        if not os.path.exists(path):
            os.makedirs(path)

        model_params = {'feature_names': self.feature_names,
                        'label_names': self.label_names,
                        'norm_params': self.normalization_parameters,
                        'model_params': self.model.model_params}

        json_path = path.rstrip('/') + '.json'
        model_params = self.dict_json_convert(model_params)
        with open(json_path, 'w') as f:
            json.dump(model_params, f, indent=2, sort_keys=True)

    @classmethod
    def build(cls, p_fun, feature_names, label_names,
              loss_weights=(0.5, 0.5), hidden_layers=None,
              metric='mae', initializer=None, optimizer=None,
              learning_rate=0.01, history=None,
              kernel_reg_rate=0.0, kernel_reg_power=1,
              bias_reg_rate=0.0, bias_reg_power=1):
        """
        Build phygnn model from given features, layers and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This function must take (y_predicted, y_true, p, **p_kwargs)
            as arguments with datatypes (tf.Tensor, np.ndarray, np.ndarray).
            The function must return a tf.Tensor object with a single numeric
            loss value (output.ndim == 0).
        feature_names : list
            Ordered list of feature names.
        label_names : list
            Ordered list of label (output) names.
        loss_weights : tuple, optional
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        hidden_layers : list, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers (9 layers total):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        initializer : tensorflow.keras.initializers, optional
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers, optional
            Instantiated neural network optimization object.
            None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).

        Returns
        -------
        PhygnnModel
            Initialized PhygnnModel instance
        """
        if isinstance(label_names, str):
            label_names = [label_names]

        model = cls.compile_model(p_fun,
                                  loss_weights=loss_weights,
                                  n_features=len(feature_names),
                                  n_labels=len(label_names),
                                  hidden_layers=hidden_layers,
                                  metric=metric,
                                  initializer=initializer,
                                  optimizer=optimizer,
                                  learning_rate=learning_rate,
                                  history=history,
                                  kernel_reg_rate=kernel_reg_rate,
                                  kernel_reg_power=kernel_reg_power,
                                  bias_reg_rate=bias_reg_rate,
                                  bias_reg_power=bias_reg_power,
                                  feature_names=feature_names,
                                  output_names=label_names)

        return cls(model, feature_names=feature_names, label_names=label_names)

    @classmethod
    def train(cls, p_fun, features, labels, p,
              loss_weights=(0.5, 0.5), hidden_layers=None,
              metric='mae', initializer=None, optimizer=None,
              learning_rate=0.01, history=None,
              kernel_reg_rate=0.0, kernel_reg_power=1,
              bias_reg_rate=0.0, bias_reg_power=1, n_batch=16, n_epoch=10,
              shuffle=True, validation_split=0.2, run_preflight=True,
              return_diagnostics=False, p_kwargs=None, parse_kwargs=None,
              norm_labels=True, save_path=None):
        """
        Build phygnn model from given features, layers and
        kwargs and then train with given labels and kwargs

        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This function must take (y_predicted, y_true, p, **p_kwargs)
            as arguments with datatypes (tf.Tensor, np.ndarray, np.ndarray).
            The function must return a tf.Tensor object with a single numeric
            loss value (output.ndim == 0).
        features : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm.
        labels : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame. If this is a
            DataFrame, the index is ignored, the columns are used with
            self.output_names, and the df is converted into a numpy array for
            batching and passing to the training algorithm.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. If this is a DataFrame, the index and column labels
            are ignored and the df is converted into a numpy array for batching
            and passing to the training algorithm and physical loss function.
        loss_weights : tuple, optional
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        hidden_layers : list, optional
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers (9 layers total):
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        initializer : tensorflow.keras.initializers, optional
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers, optional
            Instantiated neural network optimization object.
            None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate.
        history : None | pd.DataFrame, optional
            Learning history if continuing a training session.
        kernel_reg_rate : float, optional
            Kernel regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer weights and should reduce
            model complexity. Setting this to 0.0 will disable kernel
            regularization.
        kernel_reg_power : int, optional
            Kernel regularization power. kernel_reg_power=1 is L1
            regularization (lasso regression), and kernel_reg_power=2 is L2
            regularization (ridge regression).
        bias_reg_rate : float, optional
            Bias regularization rate. Increasing this value above zero will
            add a structural loss term to the loss function that
            disincentivizes large hidden layer biases and should reduce
            model complexity. Setting this to 0.0 will disable bias
            regularization.
        bias_reg_power : int, optional
            Bias regularization power. bias_reg_power=1 is L1
            regularization (lasso regression), and bias_reg_power=2 is L2
            regularization (ridge regression).
        n_batch : int
            Number of times to update the NN weights per epoch (number of
            mini-batches). The training data will be split into this many
            mini-batches and the NN will train on each mini-batch, update
            weights, then move onto the next mini-batch.
        n_epoch : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from features and labels.
        validation_split : float
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.
            Fraction of features and labels to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        parse_kwargs : dict
            kwargs for cls._parse_features
        norm_labels : bool, optional
            Flag to normalize label, by default True
        save_path : str, optional
            Directory path to save model to. The tensorflow model will be
            saved to the directory while the framework parameters will be
            saved in json, by default None

        Returns
        -------
        model : TfModel
            Initialized and trained TfModel obj
        diagnostics : dict, optional
            Namespace of training parameters that can be used for diagnostics.
        """
        _, feature_names = cls._parse_data(features)
        _, label_names = cls._parse_data(labels)

        model = cls.build(p_fun, feature_names, label_names,
                          loss_weights=loss_weights,
                          hidden_layers=hidden_layers,
                          metric=metric,
                          initializer=initializer,
                          optimizer=optimizer,
                          learning_rate=learning_rate,
                          history=history,
                          kernel_reg_rate=kernel_reg_rate,
                          kernel_reg_power=kernel_reg_power,
                          bias_reg_rate=bias_reg_rate,
                          bias_reg_power=bias_reg_power,)

        diagnostics = model.train_model(features, labels, p,
                                        n_batch=n_batch,
                                        n_epoch=n_epoch,
                                        shuffle=shuffle,
                                        validation_split=validation_split,
                                        run_preflight=run_preflight,
                                        return_diagnostics=return_diagnostics,
                                        p_kwargs=p_kwargs,
                                        parse_kwargs=parse_kwargs,
                                        norm_labels=norm_labels)

        if save_path is not None:
            model.save_model(save_path)

        if diagnostics:
            return model, diagnostics
        else:
            return model

    @classmethod
    def load(cls, path):
        """
        Load model from model path.

        Parameters
        ----------
        path : str
            Load phygnn model from pickle file.

        Returns
        -------
        model : PhygnnModel
            Loaded PhygnnModel from disk.
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

        json_path = path.rstrip('/') + '.json'
        with open(json_path, 'r') as f:
            model_params = json.load(f)

        phygnn_params = model_params.pop('model_params')
        loaded = PhysicsGuidedNeuralNetwork.set_params(phygnn_params)

        model = cls(loaded, **model_params)

        return model
