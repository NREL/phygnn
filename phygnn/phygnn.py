# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.layers import BatchNormalization, Dropout

from phygnn.utilities.loss_metrics import METRICS
from phygnn.utilities.tf_layers import Layers

logger = logging.getLogger(__name__)


def p_fun_dummy(model, y_true, y_predicted, p):
    """Example dummy function for p loss calculation.

    This dummy function does not do a real physics calculation, it just shows
    the required p_fun interface and calculates a normal MAE loss based on
    y_predicted and y_true.

    Parameters
    ----------
    model : PhysicsGuidedNeuralNetwork
        Instance of the phygnn model at the current point in training.
    y_true : np.ndarray
        Known y values that were given to the phygnn.fit() method.
    y_predicted : tf.Tensor
        Predicted y values in a 2D tensor based on x values in the
        current batch.
    p : np.ndarray
        Supplemental physical feature data that can be used to calculate a
        y_physical value to compare against y_predicted. The rows in this
        array have been carried through the batching process alongside y_true
        and the x-features used to create y_predicted and so can be used 1-to-1
        with the rows in y_predicted and y_true.

    Returns
    -------
    p_loss : tf.Tensor
        A 0D tensor physical loss value.
    """
    # pylint: disable=W0613
    return tf.math.reduce_mean(tf.math.abs(y_predicted - y_true))


class PhysicsGuidedNeuralNetwork:
    """Simple Deep Neural Network with custom physical loss function."""

    def __init__(self, p_fun, loss_weights=(0.5, 0.5),
                 n_features=1, n_labels=1, hidden_layers=None,
                 input_layer=None, output_layer=None,
                 metric='mae', initializer=None, optimizer=None,
                 learning_rate=0.01, history=None,
                 kernel_reg_rate=0.0, kernel_reg_power=1,
                 bias_reg_rate=0.0, bias_reg_power=1,
                 feature_names=None, output_names=None, name=None):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This fun must take (phygnn, y_true, y_predicted, p, **p_kwargs)
            as arguments with datatypes (PhysicsGuidedNeuralNetwork, tf.Tensor,
            np.ndarray, np.ndarray). The function must return a tf.Tensor
            object with a single numeric loss value (output.ndim == 0).
        loss_weights : tuple, optional
            Loss weights for the neural network y_true vs. y_predicted
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
        input_layer : None | InputLayer
            Keras input layer. Will default to an InputLayer with
            input shape = n_features.
        output_layer : None | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be {'units': 1, 'activation': 'sigmoid'}
            This defaults to a single dense layer with no activation
            (best for regression problems).
        metric : str, optional
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
            or a method in tensorflow.keras.losses that takes
            (y_true, y_predicted) as arguments.
        initializer : tensorflow.keras.initializers, optional
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers | dict | None
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
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
        name : None | str
            Optional model name for debugging.
        """

        self._p_fun = p_fun
        self._loss_weights = None
        self._metric = metric
        self._input_dims = n_features
        self._output_dims = n_labels
        self._layers = Layers(n_features, n_labels=n_labels,
                              hidden_layers=hidden_layers,
                              input_layer=input_layer,
                              output_layer=output_layer)
        self._optimizer = None
        self._history = history
        self._learning_rate = learning_rate
        self.kernel_reg_rate = kernel_reg_rate
        self.kernel_reg_power = kernel_reg_power
        self.bias_reg_rate = bias_reg_rate
        self.bias_reg_power = bias_reg_power
        self.feature_names = feature_names
        self.output_names = output_names
        self.name = name if isinstance(name, str) else 'phygnn'

        self.set_loss_weights(loss_weights)

        if self._metric.lower() in METRICS:
            self._metric_fun = METRICS[self._metric.lower()]
        else:
            try:
                self._metric_fun = getattr(tf.keras.losses, self._metric)
            except Exception:
                e = ('Could not recognize error metric "{}". The following '
                     'error metrics are available: {}'
                     .format(self._metric, list(METRICS.keys())))
                logger.error(e)
                raise KeyError(e)

        self._initializer = initializer
        if initializer is None:
            self._initializer = initializers.GlorotUniform()

        self._optimizer = optimizer
        if isinstance(optimizer, dict):
            class_name = optimizer['name']
            OptimizerClass = getattr(optimizers, class_name)
            self._optimizer = OptimizerClass.from_config(optimizer)
        elif optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

    @property
    def history(self):
        """
        Model training history DataFrame (None if not yet trained)

        Returns
        -------
        pandas.DataFrame | None
        """
        return self._history

    @property
    def layers(self):
        """
        TensorFlow keras layers

        Returns
        -------
        list
        """
        return self._layers.layers

    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        return self._layers.weights

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
        return self._layers.kernel_weights

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
        return self._layers.bias_weights

    @property
    def kernel_reg_term(self):
        """Get the regularization term for the kernel regularization without
        the regularization rate applied."""
        loss_k_reg = [tf.math.abs(x) for x in self.kernel_weights]
        loss_k_reg = [tf.math.pow(x, self.kernel_reg_power)
                      for x in loss_k_reg]
        loss_k_reg = tf.math.reduce_sum(
            [tf.math.reduce_sum(x) for x in loss_k_reg])

        return loss_k_reg

    @property
    def bias_reg_term(self):
        """Get the regularization term for the bias regularization without
        the regularization rate applied."""
        loss_b_reg = [tf.math.abs(x) for x in self.bias_weights]
        loss_b_reg = [tf.math.pow(x, self.bias_reg_power)
                      for x in loss_b_reg]
        loss_b_reg = tf.math.reduce_sum(
            [tf.math.reduce_sum(x) for x in loss_b_reg])

        return loss_b_reg

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        weight_dict = {}
        for i, layer in enumerate(self.layers):
            weight_dict[i] = layer.get_weights()

        model_params = {'p_fun': self._p_fun,
                        'hidden_layers': self._layers.hidden_layer_kwargs,
                        'loss_weights': self._loss_weights,
                        'metric': self._metric,
                        'n_features': self._input_dims,
                        'n_labels': self._output_dims,
                        'initializer': self._initializer,
                        'optimizer': self._optimizer.get_config(),
                        'learning_rate': self._learning_rate,
                        'weight_dict': weight_dict,
                        'history': self.history,
                        'kernel_reg_rate': self.kernel_reg_rate,
                        'kernel_reg_power': self.kernel_reg_power,
                        'bias_reg_rate': self.bias_reg_rate,
                        'bias_reg_power': self.bias_reg_power,
                        'feature_names': self.feature_names,
                        'output_names': self.output_names,
                        }

        return model_params

    @staticmethod
    def _check_shapes(x, y):
        """Check the shape of two input arrays for usage in this NN."""
        assert len(x.shape) == 2, 'Input dimensions must be 2D!'
        assert len(y.shape) == 2, 'Input dimensions must be 2D!'
        assert len(x) == len(y), 'Number of input observations dont match!'

        return True

    @staticmethod
    def seed(s=0):
        """
        Set the random seed for reproducable results.

        Parameters
        ----------
        s : int
            Random seed
        """
        np.random.seed(s)
        tf.random.set_seed(s)

    @staticmethod
    def _get_val_split(x, y, p, shuffle=True, validation_split=0.2):
        """Get a validation split and remove from from the training data.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        y : np.ndarray
            Known output data in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        shuffle : bool
            Flag to randomly subset the validation data from x and y.
            shuffle=False will take the first entries in x and y.
        validation_split : float
            Fraction of x and y to put in the validation set.

        Returns
        -------
        x : np.ndarray
            Feature data for model training as 2D array. Length of this
            will be the length of the input x multiplied by one minus
            the split fraction
        y : np.ndarray
            Known output data for model training as 2D array. Length of this
            will be the length of the input y multiplied by one minus
            the split fraction
        p : np.ndarray
            Supplemental feature data for physics loss function to be used
            in model training as 2D array. Length of this will be the length
            of the input p multiplied by one minus the split fraction
        x_val : np.ndarray
            Feature data for model validation as 2D array. Length of this
            will be the length of the input x multiplied by the split fraction
        y_val : np.ndarray
            Known output data for model validation as 2D array. Length of this
            will be the length of the input y multiplied by the split fraction
        p_val : np.ndarray
            Supplemental feature data for physics loss function to be used in
            model validation as 2D array. Length of this will be the length of
            the input p multiplied by the split fraction
        """

        L = len(x)
        n = int(L * validation_split)

        if shuffle:
            i = np.random.choice(L, replace=False, size=(n,))
        else:
            i = np.arange(n)

        j = np.array(list(set(range(L)) - set(i)))

        assert len(set(i)) == len(i)
        assert len(set(list(i) + list(j))) == L

        x_val, y_val, p_val = x[i, :], y[i, :], p[i, :]
        x, y, p = x[j, :], y[j, :], p[j, :]

        PhysicsGuidedNeuralNetwork._check_shapes(x_val, y_val)
        PhysicsGuidedNeuralNetwork._check_shapes(x_val, p_val)
        PhysicsGuidedNeuralNetwork._check_shapes(x, y)
        PhysicsGuidedNeuralNetwork._check_shapes(x, p)

        logger.debug('Validation data has length {} and training data has '
                     'length {} (split of {})'
                     .format(len(x_val), len(x), validation_split))

        return x, y, p, x_val, y_val, p_val

    @staticmethod
    def _make_batches(x, y, p, n_batch=16, shuffle=True):
        """Make lists of batches from x and y.

        Parameters
        ----------
        x : np.ndarray
            Feature data for training in a 2D array
        y : np.ndarray
            Known output data for training in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        n_batch : int
            Number of times to update the NN weights per epoch. The training
            data will be split into this many batches and the NN will train on
            each batch, update weights, then move onto the next batch.
        shuffle : bool
            Flag to randomly subset the validation data from x and y.

        Returns
        -------
        x_batches : list
            List of 2D arrays that are split subsets of x.
            Length of list is n_batch.
        y_batches : list
            List of 2D arrays that are split subsets of y.
            Length of list is n_batch.
        p_batches : list
            List of 2D arrays that are split subsets of p.
            Length of list is n_batch.
        """

        L = len(x)
        if shuffle:
            i = np.random.choice(L, replace=False, size=(L,))
            assert len(set(i)) == L
        else:
            i = np.arange(L)

        batch_indexes = np.array_split(i, n_batch)

        x_batches = [x[j, :] for j in batch_indexes]
        y_batches = [y[j, :] for j in batch_indexes]
        p_batches = [p[j, :] for j in batch_indexes]

        return x_batches, y_batches, p_batches

    def preflight_p_fun(self, x, y_true, p, p_kwargs):
        """Run a pre-flight check making sure the p_fun is differentiable."""

        if p_kwargs is None:
            p_kwargs = {}

        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False)
            p_loss = self._p_fun(self, y_true, y_predicted, p, **p_kwargs)
            grad = tape.gradient(p_loss, self.weights)

            if not tf.is_tensor(p_loss):
                emsg = 'Loss output from p_fun() must be a tensor!'
                logger.error(emsg)
                raise TypeError(emsg)

            if p_loss.ndim > 1:
                emsg = ('Loss output from p_fun() should be a scalar tensor '
                        'but received a tensor with shape {}'
                        .format(p_loss.shape))
                logger.error(emsg)
                raise ValueError(emsg)

            assert isinstance(grad, list)
            if grad[0] is None:
                emsg = ('The input p_fun was not differentiable! '
                        'Please use only tensor math in the p_fun.')
                logger.error(emsg)
                raise RuntimeError(emsg)

        logger.debug('p_fun passed preflight check.')

    def preflight_data(self, x, y, p):
        """Run simple preflight checks on data shapes.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm.
        y : np.ndarray | pd.DataFrame
            Known output data in a 2D array or DataFrame. If this is a
            DataFrame, the index is ignored, the columns are used with
            self.output_names, and the df is converted into a numpy array for
            batching and passing to the training algorithm.
        p : np.ndarray | pd.DataFrame
            Supplemental feature data for the physics loss function in 2D array
            or DataFrame. If this is a DataFrame, the index and column labels
            are ignored and the df is converted into a numpy array for batching
            and passing to the training algorithm and physical loss function.

        Returns
        ----------
        x : np.ndarray
            Feature data in a 2D array
        y : np.ndarray
            Known output data in a 2D array
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        """

        self._check_shapes(x, y)
        self._check_shapes(x, p)
        x_msg = ('x data has {} features but expected {}'
                 .format(x.shape[1], self._input_dims))
        y_msg = ('y data has {} features but expected {}'
                 .format(y.shape[1], self._output_dims))
        assert x.shape[1] == self._input_dims, x_msg
        assert y.shape[1] == self._output_dims, y_msg

        x = self.preflight_features(x)

        if isinstance(y, pd.DataFrame):
            y_cols = y.columns.values.tolist()
            if self.output_names is None:
                self.output_names = y_cols
            else:
                msg = ('Cannot work with input y columns: {}, previously set '
                       'output names are: {}'
                       .format(y_cols, self.output_names))
                assert self.output_names == y_cols, msg
            y = y.values

        if isinstance(p, pd.DataFrame):
            p = p.values

        return x, y, p

    def preflight_features(self, x):
        """Run preflight checks and data conversions on feature data.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm.

        Returns
        ----------
        x : np.ndarray
            Feature data in a 2D array
        """

        assert len(x.shape) == 2, 'PhyGNN can only use 2D data as input!'
        x_msg = ('x data has {} features but expected {}'
                 .format(x.shape[1], self._input_dims))
        assert x.shape[1] == self._input_dims, x_msg

        if isinstance(x, pd.DataFrame):
            x_cols = x.columns.values.tolist()
            if self.feature_names is None:
                self.feature_names = x_cols
            else:
                msg = ('Cannot work with input x columns: {}, previously set '
                       'feature names are: {}'
                       .format(x_cols, self.feature_names))
                assert self.feature_names == x_cols, msg
            x = x.values

        return x

    def reset_history(self):
        """Erase previous training history without resetting trained weights"""
        self._history = None

    def set_loss_weights(self, loss_weights):
        """Set new loss weights

        Parameters
        ----------
        loss_weights : tuple
            Loss weights for the neural network y_true vs y_predicted
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the phygnn loss function
            to just the p_fun output.
        """
        assert np.sum(loss_weights) > 0, 'Sum of loss_weights must be > 0!'
        assert len(loss_weights) == 2, 'loss_weights can only have two values!'
        self._loss_weights = loss_weights

    def loss(self, y_true, y_predicted, p, p_kwargs):
        """Calculate the loss function by comparing y_true to model-predicted y

        Parameters
        ----------
        y_true : np.ndarray
            Known output data in a 2D array.
        y_predicted : tf.Tensor
            Model-predicted output data in a 2D tensor.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.

        Returns
        -------
        loss : tf.tensor
            Sum of the NN loss function comparing the y_predicted against
            y_true and the physical loss function (self._p_fun) with
            respective weights applied.
        nn_loss : tf.tensor
            Standard NN training loss comparing y to y_predicted.
        p_loss : tf.tensor
            Physics loss from p_fun.
        """

        if p_kwargs is None:
            p_kwargs = {}

        loss = tf.constant(0.0, dtype=tf.float32)
        nn_loss = tf.constant(0.0, dtype=tf.float32)
        p_loss = tf.constant(0.0, dtype=tf.float32)

        if self._loss_weights[0] != 0:
            nn_loss = self._metric_fun(y_true, y_predicted)
            msg = ('Bad shape from nn_loss fun! Must be 0D but received: {}'
                   .format(nn_loss))
            assert nn_loss.ndim == 0, msg
            loss += self._loss_weights[0] * nn_loss

        if self._loss_weights[1] != 0:
            p_loss = self._p_fun(self, y_true, y_predicted, p, **p_kwargs)
            msg = ('Bad shape from p_loss fun! Must be 0D but received: {}'
                   .format(p_loss))
            assert p_loss.ndim == 0, msg
            loss += self._loss_weights[1] * p_loss

        logger.debug('NN Loss: {:.2e}, P Loss: {:.2e}, Total Loss: {:.2e}'
                     .format(nn_loss, p_loss, loss))

        if self.kernel_reg_rate != 0:
            loss_kernel_reg = self.kernel_reg_term * self.kernel_reg_rate
            loss += loss_kernel_reg
            logger.debug('Kernel regularization loss: {:.2e}, '
                         'Total Loss: {:.2e}'.format(loss_kernel_reg, loss))

        if self.bias_reg_rate != 0:
            loss_bias_reg = self.bias_reg_term * self.bias_reg_rate
            loss += loss_bias_reg
            logger.debug('Bias regularization loss: {:.2e}, '
                         'Total Loss: {:.2e}'.format(loss_bias_reg, loss))

        if tf.math.is_nan(loss):
            msg = 'phygnn calculated a NaN loss value!'
            logger.error(msg)
            raise ArithmeticError(msg)

        return loss, nn_loss, p_loss

    def _get_grad(self, x, y_true, p, p_kwargs):
        """Get the gradient based on a mini-batch of x and y_true data."""
        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False, training=True)
            loss = self.loss(y_true, y_predicted, p, p_kwargs)[0]
            grad = tape.gradient(loss, self.weights)

        return grad, loss

    def _run_gradient_descent(self, x, y_true, p, p_kwargs):
        """Run gradient descent for one mini-batch of (x, y_true)
        and adjust NN weights."""
        grad, loss = self._get_grad(x, y_true, p, p_kwargs)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return grad, loss

    def fit(self, x, y, p, n_batch=16, n_epoch=10, shuffle=True,
            validation_split=0.2, p_kwargs=None, run_preflight=True,
            return_diagnostics=False):
        """Fit the neural network to data from x and y.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Feature data in a 2D array or DataFrame. If this is a DataFrame,
            the index is ignored, the columns are used with self.feature_names,
            and the df is converted into a numpy array for batching and passing
            to the training algorithm.
        y : np.ndarray | pd.DataFrame
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
            from x and y.
        validation_split : float
            Fraction of x and y to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        run_preflight : bool
            Flag to run preflight checks.
        return_diagnostics : bool
            Flag to return training diagnostics dictionary.

        Returns
        -------
        diagnostics : dict
            Namespace of training parameters that can be used for diagnostics.
        """

        x, y, p = self.preflight_data(x, y, p)

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(
                columns=['elapsed_time', 'training_loss', 'validation_loss'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        x, y, p, x_val, y_val, p_val = self._get_val_split(
            x, y, p, shuffle=shuffle, validation_split=validation_split)

        if self._loss_weights[1] > 0 and run_preflight:
            self.preflight_p_fun(x_val, y_val, p_val, p_kwargs)

        t0 = time.time()
        for epoch in epochs:

            x_batches, y_batches, p_batches = self._make_batches(
                x, y, p, n_batch=n_batch, shuffle=shuffle)

            batch_iter = zip(x_batches, y_batches, p_batches)
            for x_batch, y_batch, p_batch in batch_iter:
                tr_loss = self._run_gradient_descent(
                    x_batch, y_batch, p_batch, p_kwargs)[1]

            y_val_pred = self.predict(x_val, to_numpy=False)
            val_loss = self.loss(y_val, y_val_pred, p_val, p_kwargs)[0]
            logger.info('Epoch {} train loss: {:.2e} '
                        'val loss: {:.2e} for "{}"'
                        .format(epoch, tr_loss, val_loss, self.name))

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = tr_loss.numpy()
            self._history.at[epoch, 'validation_loss'] = val_loss.numpy()

        diagnostics = {'x': x, 'y': y, 'p': p,
                       'x_val': x_val, 'y_val': y_val, 'p_val': p_val,
                       'history': self.history,
                       }

        if return_diagnostics:
            return diagnostics

    def predict(self, x, to_numpy=True, training=False):
        """Run a prediction on input features.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        to_numpy : bool
            Flag to convert output from tensor to numpy array
        training : bool
            Flag for predict() used in the training routine. This is used
            to freeze the BatchNormalization and Dropout layers.

        Returns
        -------
        y : tf.Tensor | np.ndarray
            Predicted output data in a 2D array.
        """

        x = self.preflight_features(x)

        # run x through the input layer to get y
        y = self.layers[0](x)

        for layer in self.layers[1:]:
            if isinstance(layer, (BatchNormalization, Dropout)):
                y = layer(y, training=training)
            else:
                y = layer(y)

        if to_numpy:
            y = y.numpy()

        return y

    def save(self, fpath):
        """Save phygnn model to pickle file.

        Parameters
        ----------
        fpath : str
            File path to .pkl file to save model to.
        """

        if not fpath.endswith('.pkl'):
            e = 'Can only save model to .pkl file!'
            logger.error(e)
            raise ValueError(e)

        dirname = os.path.dirname(fpath)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        model_params = self.model_params

        with open(fpath, 'wb') as f:
            pickle.dump(model_params, f)

    @classmethod
    def set_params(cls, model_params):
        """
        Initialize phygnn model from saved model parameters

        Parameters
        ----------
        model_params : dict
            Model parameters

        Returns
        -------
        model : PhysicsGuidedNeuralNetwork
            Instantiated phygnn model
        """
        p_fun = model_params.pop('p_fun')
        weight_dict = model_params.pop('weight_dict')

        model = cls(p_fun, **model_params)

        for i, weights in weight_dict.items():
            if weights:
                dim = weights[0].shape[0]

                if isinstance(model.layers[i], BatchNormalization):
                    # BatchNormalization layers need to be
                    # built with funky input dims.
                    model.layers[i].build((None, dim))
                else:
                    model.layers[i].build((dim,))

                model.layers[i].set_weights(weights)

        return model

    @classmethod
    def load(cls, fpath):
        """Load a phygnn model that has been saved to a pickle file.

        Parameters
        ----------
        fpath : str
            File path to .pkl file to load model from.

        Returns
        -------
        model : PhysicsGuidedNeuralNetwork
            Instantiated phygnn model
        """
        if not os.path.exists(fpath):
            e = 'Could not load file, does not exist: {}'.format(fpath)
            logger.error(e)
            raise FileNotFoundError(e)

        if not fpath.endswith('.pkl'):
            e = 'Can only load model from .pkl file!'
            logger.error(e)
            raise ValueError(e)

        with open(fpath, 'rb') as f:
            model_params = pickle.load(f)

        model = cls.set_params(model_params)

        return model
