# -*- coding: utf-8 -*-
"""
Tensorflow Layers Handlers
"""
import copy
from tensorflow.keras.layers import (InputLayer, Dense, Dropout, Activation,
                                     BatchNormalization)


class HiddenLayers:
    """
    Class to handle TensorFlow hidden layers
    """

    def __init__(self, hidden_layers):
        """
        Parameters
        ----------
        hidden_layers : list
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers:
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]
        """
        self._layers = []
        self._hidden_layers = copy.deepcopy(hidden_layers)
        for layer in hidden_layers:
            self.add_layer(layer)

        self._i = 0

    def __repr__(self):
        msg = "{} with {} hidden layers".format(self.__class__.__name__,
                                                len(self))

        return msg

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        """
        Get layer at given index

        Parameters
        ----------
        idx : int
            index of layer to extract

        Returns
        -------
        tensorflow.keras.layers
        """
        return self.layers[idx]

    def __setitem__(self, idx, layer_kwargs):
        """
        Add layer

        Parameters
        ----------
        idx : int | None
            Index to insert layer at, if None append to the end
        layer_kwargs : dict
            Dictionary of key word arguments for list layer. For example,
            any of the following are valid inputs:
                {'units': 64, 'activation': 'relu', 'dropout': 0.05}
                {'units': 64, 'name': 'relu1'}
                {'activation': 'relu'}
                {'batch_normalization': {'axis': -1}}
                {'dropout': 0.1}
        """
        self.add_layer(layer_kwargs, insert_index=idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self):
            self._i = 0
            raise StopIteration

        layer = self[self._i]
        self._i += 1

        return layer

    @property
    def layers(self):
        """
        TensorFlow keras layers

        Returns
        -------
        list
        """
        return self._layers

    @property
    def hidden_layer_kwargs(self):
        """
        List of dictionaries of key word arguments for each hidden
        layer in the NN. Dense linear layers can be input with their
        activations or separately for more explicit control over the layer
        ordering. For example, this is a valid input for hidden_layers that
        will yield 7 hidden layers:
            [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                {'units': 64},
                {'batch_normalization': {'axis': -1}},
                {'activation': 'relu'},
                {'dropout': 0.01}]

        Returns
        -------
        list
        """
        return self._hidden_layers

    @property
    def weights(self):
        """
        Get a list of layer weights for gradient calculations.

        Returns
        -------
        list
        """
        weights = []
        for layer in self:
            # Include gamma and beta weights for BatchNormalization
            # but do not include moving mean/stdev
            if isinstance(layer, BatchNormalization):
                weights += layer.variables[:2]

            elif layer.trainable:
                weights += layer.trainable_weights

        return weights

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
        weights = []
        for layer in self:
            if isinstance(layer, Dense):
                weights.append(layer.variables[0])

        return weights

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
        weights = []
        for layer in self:
            if isinstance(layer, Dense):
                weights.append(layer.variables[1])

        return weights

    def add_layer(self, layer_kwargs, insert_index=None):
        """Add a hidden layer to the DNN.

        Parameters
        ----------
        layer_kwargs : dict
            Dictionary of key word arguments for list layer. For example,
            any of the following are valid inputs:
                {'units': 64, 'activation': 'relu', 'dropout': 0.05}
                {'units': 64, 'name': 'relu1'}
                {'activation': 'relu'}
                {'batch_normalization': {'axis': -1}}
                {'dropout': 0.1}
        insert_index : int | None
            Optional index to insert the new layer at. None will append
            the layer to the end of the layer list.
        """

        layer_kwargs_cp = copy.deepcopy(layer_kwargs)
        dense_layer = None
        bn_layer = None
        a_layer = None
        d_layer = None
        activation_arg = layer_kwargs_cp.pop('activation', None)
        dropout_rate = layer_kwargs_cp.pop('dropout', None)
        batch_norm_kwargs = layer_kwargs_cp.pop('batch_normalization', None)

        if 'units' in layer_kwargs_cp:
            dense_layer = Dense(**layer_kwargs_cp)
            if insert_index is not None:
                self._layers.insert(insert_index, dense_layer)
            else:
                self._layers.append(dense_layer)

        if batch_norm_kwargs is not None:
            bn_layer = BatchNormalization(**batch_norm_kwargs)
        if activation_arg is not None:
            a_layer = Activation(activation_arg)
        if dropout_rate is not None:
            d_layer = Dropout(dropout_rate)

        # This ensures proper default ordering of layers if requested together.
        for layer in [bn_layer, a_layer, d_layer]:
            if layer is not None:
                if insert_index is not None:
                    if dense_layer is not None:
                        insert_index += 1
                    self._layers.insert(insert_index, layer)
                else:
                    self._layers.append(layer)

    @classmethod
    def compile(cls, model, hidden_layers):
        """
        Add hidden layers to model

        Parameters
        ----------
        model : tensorflow.keras
            Model to add hidden layers too
        hidden_layers : list
            List of dictionaries of key word arguments for each hidden
            layer in the NN. Dense linear layers can be input with their
            activations or separately for more explicit control over the layer
            ordering. For example, this is a valid input for hidden_layers that
            will yield 7 hidden layers:
                [{'units': 64, 'activation': 'relu', 'dropout': 0.01},
                 {'units': 64},
                 {'batch_normalization': {'axis': -1}},
                 {'activation': 'relu'},
                 {'dropout': 0.01}]

        Returns
        -------
        model : tensorflow.keras
            Model with layers added
        """
        hidden_layers = cls(hidden_layers)
        for layer in hidden_layers:
            model.add(layer)

        return model


class Layers(HiddenLayers):
    """
    Class to handle TensorFlow layers
    """

    def __init__(self, n_features, n_labels=1, hidden_layers=None,
                 input_layer=None, output_layer=None):
        """
        Parameters
        ----------
        n_features : int
            Number of features (inputs) to train the model on
        n_labels : int, optional
            Number of labels (outputs) to the model, by default 1
        hidden_layers : list | None, optional
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
            by default None which will lead to a single linear layer
        input_layer : None | InputLayer
            Keras input layer. Will default to an InputLayer with
            input shape = n_features.
        output_layer : None | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}]
            This defaults to a single dense layer with no activation
            (best for regression problems).
        """

        self._layers = input_layer
        if input_layer is None:
            self._layers = [InputLayer(input_shape=[n_features])]

        self._hidden_layers = copy.deepcopy(hidden_layers)
        if hidden_layers:
            for layer in hidden_layers:
                self.add_layer(layer)

        if output_layer is None:
            self._layers.append(Dense(n_labels))
        else:
            if not isinstance(output_layer, list):
                output_layer = [output_layer]
            for layer in output_layer:
                self.add_layer(layer)

        self._i = 0

    @classmethod
    def compile(cls, model, n_features, n_labels=1, hidden_layers=None,
                input_layer=None, output_layer=None):
        """
        Build all layers needed for model

        Parameters
        ----------
        model : tensorflow.keras | PhysicsGuidedNeuralNetwork
            Model to add layers too
        n_features : int
            Number of features (inputs) to train the model on
        n_labels : int, optional
            Number of labels (outputs) to the model, by default 1
        hidden_layers : list | None, optional
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
            by default None which will lead to a single linear layer
        input_layer : None | InputLayer
            Keras input layer. Will default to an InputLayer with
            input shape = n_features.
        output_layer : None | list | dict
            Output layer specification. Can be a list/dict similar to
            hidden_layers input specifying a dense layer with activation.
            For example, for a classfication problem with a single output,
            output_layer should be [{'units': 1}, {'activation': 'sigmoid'}]
            This defaults to a single dense layer with no activation
            (best for regression problems).

        Returns
        -------
        model : tensorflow.keras
            Model with layers added
        """
        layers = cls(n_features, n_labels=n_labels,
                     hidden_layers=hidden_layers,
                     input_layer=input_layer,
                     output_layer=output_layer)
        for layer in layers:
            model.add(layer)

        return model
