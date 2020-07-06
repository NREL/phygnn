# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""
import os
import pickle
import copy
import time
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras import layers, optimizers, initializers

from phygnn.loss_metrics import METRICS


logger = logging.getLogger(__name__)


class PhysicsGuidedNeuralNetwork:
    """Simple Deep Neural Network with custom physical loss function."""

    def __init__(self, p_fun, hidden_layers, loss_weights=(0.5, 0.5),
                 input_dims=1, output_dims=1, metric='mae',
                 initializer=None, optimizer=None,
                 learning_rate=0.01, history=None):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This function must take (y_predicted, y_true, p, **p_kwargs)
            as arguments with datatypes (tf.Tensor, np.ndarray, np.ndarray).
            The function must return a tf.Tensor object with a single numeric
            loss value (output.ndim == 0).
        hidden_layers : list
            List of dictionaries of key word arguments for each hidden
            layer in the NN. For example:
            hidden_layers=[{'units': 64, 'activation': 'relu',
                            'name': 'layer1', 'dropout': 0.01},
                           {'units': 64, 'activation': 'relu',
                            'name': 'layer2', 'dropout': 0.01}]
        loss_weights : tuple
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the PGNN loss function
            to just the p_fun output.
        input_dims : int
            Number of input features.
        output_dims : int
            Number of output labels.
        metric : str
            Loss metric option for the NN loss function (not the physical
            loss function). Must be a valid key in phygnn.loss_metrics.METRICS
        initializer : tensorflow.keras.initializers
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers
            Instantiated neural network optimization object.
            None defaults to Adam.
        learning_rate : float
            Optimizer learning rate.
        history : None | pd.dataframe
            Learning history if continuing a training session.
        """

        self._p_fun = p_fun
        self._hidden_layers = copy.deepcopy(hidden_layers)
        self._loss_weights = None
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._layers = []
        self._optimizer = None
        self._history = history
        self._learning_rate = learning_rate

        self.set_loss_weights(loss_weights)

        if metric.lower() not in METRICS:
            e = ('Could not recognize error metric "{}". The following error '
                 'metrics are available: {}'
                 .format(metric, list(METRICS.keys())))
            logger.error(e)
            raise KeyError(e)
        else:
            self._metric_fun = METRICS[metric.lower()]

        self._initializer = initializer
        if initializer is None:
            self._initializer = initializers.GlorotUniform()

        self._optimizer = optimizer
        if optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)

        self._layers.append(layers.InputLayer(input_shape=[input_dims]))
        for hidden_layer in hidden_layers:
            self.add_layer(hidden_layer)
        self._layers.append(layers.Dense(
            output_dims, kernel_initializer=self._initializer))

    @staticmethod
    def _check_shapes(x, y):
        """Check the shape of two input arrays for usage in this NN."""
        assert len(x.shape) == 2, 'Input dimensions must be 2D!'
        assert len(y.shape) == 2, 'Input dimensions must be 2D!'
        assert len(x) == len(y), 'Number of input observations dont match!'
        return True

    @staticmethod
    def seed(s=0):
        """Set the random seed for reproducable results."""
        np.random.seed(s)
        tf.random.set_seed(s)

    @property
    def history(self):
        """Get the training history dataframe (None if not yet trained)."""
        return self._history

    @property
    def layers(self):
        """Get a list of the NN layers."""
        return self._layers

    @property
    def weights(self):
        """Get a list of layer weights for gradient calculations."""
        weights = []
        for layer in self._layers:
            weights += layer.variables
        return weights

    def reset_history(self):
        """Erase previous training history without resetting trained weights"""
        self._history = None

    def set_loss_weights(self, loss_weights):
        """Set new loss weights

        Parameters
        ----------
        loss_weights : tuple
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the PGNN loss function
            to just the p_fun output.
        """
        assert np.sum(loss_weights) > 0, 'Sum of loss_weights must be > 0!'
        assert len(loss_weights) == 2, 'loss_weights can only have two values!'
        self._loss_weights = loss_weights

    def loss(self, y_predicted, y_true, p, p_kwargs):
        """Calculate the loss function by comparing model-predicted y to y_true

        Parameters
        ----------
        y_predicted : tf.Tensor
            Model-predicted output data in a 2D tensor.
        y_true : np.ndarray
            Known output data in a 2D array.
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
        """

        if p_kwargs is None:
            p_kwargs = {}

        nn_loss = self._metric_fun(y_predicted, y_true)
        p_loss = self._p_fun(y_predicted, y_true, p, **p_kwargs)

        loss = self._loss_weights[0] * nn_loss
        if not tf.math.is_nan(p_loss) and self._loss_weights[1] > 0:
            loss += self._loss_weights[1] * p_loss

        logger.debug('NN Loss: {:.2e}, P Loss: {:.2e}, Total Loss: {:.2e}'
                     .format(nn_loss, p_loss, loss))

        return loss, nn_loss, p_loss

    def add_layer(self, layer_kwargs, insert_index=None):
        """Add a hidden layer to the DNN.

        Parameters
        ----------
        layer_kwargs : dict
            Dictionary of key word arguments for list layer. For example:
            layer_kwargs={'units': 64, 'activation': 'relu',
                          'name': 'relu1', 'dropout': 0.01}
        insert_index : int | None
            Optional index to insert the new layer at. None will append
            the layer to the end of the layer list.
        """

        dropout = layer_kwargs.pop('dropout', None)
        layer = layers.Dense(**layer_kwargs)
        if insert_index:
            self._layers.insert(insert_index, layer)
        else:
            self._layers.append(layer)

        if dropout is not None:
            d_layer = layers.Dropout(dropout)
            if insert_index:
                self._layers.insert(insert_index + 1, d_layer)
            else:
                self._layers.append(d_layer)

    def _get_grad(self, x, y_true, p, p_kwargs):
        """Get the gradient based on a mini-batch of x and y_true data."""
        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False)
            loss = self.loss(y_predicted, y_true, p, p_kwargs)[0]
            grad = tape.gradient(loss, self.weights)

        return grad, loss

    def _run_sgd(self, x, y_true, p, p_kwargs):
        """Run stochastic gradient descent for one mini-batch of (x, y_true)
        and adjust NN weights."""
        grad, loss = self._get_grad(x, y_true, p, p_kwargs)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return grad, loss

    def _p_fun_preflight(self, x, y_true, p, p_kwargs):
        """Run a pre-flight check making sure the p_fun is differentiable."""

        if p_kwargs is None:
            p_kwargs = {}

        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False)
            p_loss = self._p_fun(y_predicted, y_true, p, **p_kwargs)
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

    def fit(self, x, y, p, n_batch=16, n_epoch=10, shuffle=True,
            validation_split=0.2, p_kwargs=None, run_preflight=True,
            return_diagnostics=False):
        """Fit the neural network to data from x and y.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        y : np.ndarray
            Known output data in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
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

        self._check_shapes(x, y)
        self._check_shapes(x, p)

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
            self._p_fun_preflight(x_val, y_val, p_val, p_kwargs)

        t0 = time.time()
        for epoch in epochs:

            x_batches, y_batches, p_batches = self._make_batches(
                x, y, p, n_batch=n_batch, shuffle=shuffle)

            batch_iter = zip(x_batches, y_batches, p_batches)
            for x_batch, y_batch, p_batch in batch_iter:
                tr_loss = self._run_sgd(x_batch, y_batch, p_batch, p_kwargs)[1]

            y_val_pred = self.predict(x_val, to_numpy=False)
            val_loss = self.loss(y_val_pred, y_val, p_val, p_kwargs)[0]
            logger.info('Epoch {} training loss: {:.2e} '
                        'validation loss: {:.2e}'
                        .format(epoch, tr_loss, val_loss))

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = tr_loss.numpy()
            self._history.at[epoch, 'validation_loss'] = val_loss.numpy()

        diagnostics = {'x': x, 'y': y, 'p': p,
                       'x_val': x_val, 'y_val': y_val, 'p_val': p_val,
                       'history': self.history,
                       }

        if return_diagnostics:
            return diagnostics

    def predict(self, x, to_numpy=True):
        """Run a prediction on input features.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        to_numpy : bool
            Flag to convert output from tensor to numpy array

        Returns
        -------
        y : tf.Tensor | np.ndarray
            Predicted output data in a 2D array.
        """

        y = self._layers[0](x)
        for layer in self._layers[1:]:
            y = layer(y)

        if to_numpy:
            y = y.numpy()

        return y

    def save(self, fpath):
        """Save pgnn model to pickle file.

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

        weight_dict = {}
        for i, layer in enumerate(self._layers):
            weight_dict[i] = layer.get_weights()

        model_params = {'p_fun': self._p_fun,
                        'hidden_layers': self._hidden_layers,
                        'loss_weights': self._loss_weights,
                        'input_dims': self._input_dims,
                        'output_dims': self._output_dims,
                        'initializer': self._initializer,
                        'optimizer': self._optimizer,
                        'learning_rate': self._learning_rate,
                        'weight_dict': weight_dict,
                        'history': self._history,
                        }

        with open(fpath, 'wb') as f:
            pickle.dump(model_params, f)

    @classmethod
    def load(cls, fpath):
        """Load a pgnn model that has been saved to a pickle file.

        Parameters
        ----------
        fpath : str
            File path to .pkl file to load model from.
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

        weight_dict = model_params.pop('weight_dict')

        model = cls(**model_params)

        for i, weights in weight_dict.items():
            if weights:
                dim = weights[0].shape[0]
                model._layers[i].build((dim,))
                model._layers[i].set_weights(weights)

        return model
