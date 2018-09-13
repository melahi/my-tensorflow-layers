import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers


class Relation(tf.keras.layers.Layer):
    def __init__(self,
                 relations,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.relations = relations
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_1 = None
        self.kernel_2 = None
        self.kernel_g = None
        self.bias = None

    def build(self, input_shape):
        # The input_shape should be [batch_size, sequence_length, channels]
        # TODO: perhaps it's not bad to use InputSec to check and verify input_shape according to our assumption.

        input_shape = tf.TensorShape(input_shape)
        kernel_shape = [input_shape[-1].value, self.relations]
        self.kernel_1 = self.add_weight('kernel_1',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_2 = self.add_weight('kernel_2',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_g = self.add_weight('kernel_g',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.bias = self.add_weight('bias',
                                    shape=[self.relations],
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    dtype=self.dtype,
                                    trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        z1 = tf.tensordot(inputs, self.kernel_1, axes=[[-1], [0]])
        z1 = tf.nn.bias_add(z1, self.bias)
        z2 = tf.tensordot(inputs, self.kernel_2, axes=[[-1], [0]])

        z1 = tf.reshape(z1, shape=[-1, inputs.shape[1].value, 1, self.relations])
        z2 = tf.reshape(z2, shape=[-1, 1, inputs.shape[1].value, self.relations])
        relation_function = z1 + z2
        g = tf.tensordot(inputs, self.kernel_g, axes=[[-1], [0]])
        g = tf.reshape(g, shape=[-1, 1, inputs.shape[1].value, self.relations])
        outputs = tf.multiply(relation_function, g)
        outputs = tf.reduce_sum(outputs, axis=2)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape.ndims != 3:
            raise ValueError('The input_shape must be a rank 3 tensor but we saw: %s' % input_shape)
        if input_shape[-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)

        output_shape = input_shape[:-1].concatenate(self.relations)
        return output_shape

    def get_config(self):
        config = {
            'relations': self.relations,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
