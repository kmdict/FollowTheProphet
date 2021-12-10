# -*- coding: utf-8 -*-
"""Library for building tensorflow keras model."""

import typing

import numpy as np
import tensorflow as tf

keras = tf.keras
kb = keras.backend
if typing.TYPE_CHECKING:
    from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import


class InputExtension(object):
    """Wrap an input with additional information."""

    def __init__(self,
                 name: 'str',
                 shape: 'Sequence[int]',
                 dtype: 'Union[Type[np.generic], str]',
                 categorical_size: 'int',
                 default_embedding_size: 'Union[int, None]',
                 ):
        """

        Args:
            name: name of keras input.
            shape: shape of keras input (without batch size dimension).
            dtype: dtype of keras input
            categorical_size: int.
                >0: maximum ID for a categorical input (for embedding);
                0:  for a numerical input;
                -1: for a categorical input (that does not need embedding).
            default_embedding_size: this value can be overwritten in to_embedding_layer()
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.categorical_size = categorical_size
        self.default_embedding_size = default_embedding_size
        self._keras_input: 'Union[keras.layers.Input, None]' = None

    @property
    def keras_input(self) -> 'keras.layers.Input':
        """Get the keras input. Always return the same single instance if being called multiple times."""
        if self._keras_input is None:
            self._keras_input = keras.layers.Input(
                shape=self.shape,
                name=self.name,
                dtype=self.dtype,
            )
        return self._keras_input

    def to_embedding_layer(self,
                           embedding_size: 'Union[int, None]' = None,
                           name_pattern: 'str' = 'emb_{input_name}',
                           layer_kwargs: 'Union[Dict[str, Any], None]' = None,
                           ) -> 'keras.layers.Layer':
        """Generate a embedding layer for the input.
        For categorical input with categorical_size>0, generate a keras embedding layer.
        For numerical input (categorical_size==0), generate a dense layer.
        For categorical input with categorical_size==-1, generate a one-hot layer.

        Args:
            embedding_size: if not None, overwrite default_embedding_size.
            name_pattern: format string for the layer name (with '{input_name}')
            layer_kwargs: kwargs to pass to the embedding layer.

        Returns:
            A keras layer with output shape (batch_size, embedding_size,)
        """
        if layer_kwargs is None:
            layer_kwargs = {}
        if embedding_size is None:
            embedding_size = self.default_embedding_size
        assert embedding_size is not None and embedding_size > 0, f'Error embedding_size: {embedding_size}'

        if self.categorical_size > 0:  # categorical embedding
            if len(self.shape) == 0:
                input_length = None
            elif len(self.shape) == 1:
                input_length = self.shape[0]
            else:
                raise ValueError(f'Unsupported categorical input shape: {self.shape}')
            embedding_layer = keras.layers.Embedding(
                input_dim=self.categorical_size,
                output_dim=embedding_size,
                input_length=input_length,
                name=name_pattern.format(input_name=self.name),
                **layer_kwargs,
            )
            if input_length is not None:
                return keras.models.Sequential(
                    layers=[
                        embedding_layer,
                        keras.layers.Lambda(
                            function=kb.mean,
                            arguments=dict(axis=-2, keepdims=False),
                            name=embedding_layer.name + '_Post_mean'
                        ),
                    ],
                    name=embedding_layer.name + '_Pack'
                )
            return embedding_layer

        if self.categorical_size == 0:  # numeric embedding
            dense_layer = keras.layers.Dense(
                input_shape=self.shape,
                units=embedding_size,
                activation=None,
                use_bias=False,
                name=name_pattern.format(input_name=self.name),
                **layer_kwargs,
            )
            if len(self.shape) == 0:
                return keras.models.Sequential(
                    layers=[
                        keras.layers.Lambda(
                            function=kb.expand_dims,
                            arguments=dict(axis=-1),
                            name=dense_layer.name + '_Pre_expand_dims'
                        ),
                        dense_layer,
                    ],
                    name=dense_layer.name + '_Pack'
                )
            return dense_layer

        if self.categorical_size == -1:  # one-hot with dimension=embedding_size
            if embedding_size >= np.iinfo(np.int32).max - 1:
                index_dtype = tf.int64
            else:
                index_dtype = tf.int32
            base_name = name_pattern.format(input_name=self.name)
            return keras.models.Sequential(
                layers=[
                    keras.layers.Lambda(
                        function=kb.cast, arguments=dict(dtype=index_dtype), name=base_name + '_Pre_cast'
                    ),
                    keras.layers.Lambda(
                        function=kb.one_hot, arguments=dict(num_classes=embedding_size), name=base_name
                    )
                ],
                name=base_name + '_Pack'
            )

        raise ValueError(f'categorical_size={self.categorical_size}, embedding_size={embedding_size}')


LayersClassConfigType = 'Sequence[Tuple[Type[keras.layers.Layer], Mapping[str, Any]]]'  # pylint: disable=invalid-name


def build_sequential(layers_class_config: 'LayersClassConfigType',
                     name_prefix: 'Union[str, None]',
                     name: 'Union[str, None]'
                     ) -> 'keras.models.Sequential':
    seq = keras.models.Sequential(name=name)
    for layer_class, layer_config in layers_class_config:
        if name_prefix is not None and layer_config.get('name', None) is not None:
            layer_name = name_prefix + layer_config['name']
            layer_config = {k: v for k, v in layer_config.items() if k != 'name'}
        else:
            layer_name = layer_config.get('name', None)
        seq.add(layer_class(name=layer_name, **layer_config))
    return seq


class FnLoss(keras.losses.Loss):
    """Keras loss with a custom function."""

    def __init__(self, fn: 'Callable[[tf.Tensor, tf.Tensor], tf.Tensor]', **kwargs):
        super().__init__(**kwargs)
        self._fn = fn

    def call(self, y_true, y_pred):
        return self._fn(y_true, y_pred)
