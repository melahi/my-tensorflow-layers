import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers

def create_variable(name,
                    shape=None,
                    dtype=None,
                    initializer=None,
                    regularizer=None,
                    constraint=None,
                    trainable=True):
    return tf.get_variabl(name,
                          shape=shape,
                          dtype=dtype,
                          initializer=initializer,
                          regularizer=regularizer,
                          constraint=constraint,
                          trainable=trainable)


def new_edge(units, activation, vertices, source_edges, destination_edges, edges, universe):
    edge_weight = create_variable("edge", shape = [edges.shape[-1].value, units])
    source_weight = create_variable("source_edge", shape = [vertices.shape[-1].value, units])
    destination_weight = create_variable("destination_edge", shape = [vertices.shape[-1].value, units])
    universe_weight = create_variable("universe_edge", shape = [edges.shape[-1].value, units])
    edge_bias = create_variable("edge_bias", shape=[units])
    new_edges = tf.tensordot(edges, edges_weight, axes=[[-1], [0]])
    source_edges_vertices = tf.gather(vertices, source_edges, axis=-2)
    destination_edges_vertices = tf.gather(vertices, destination_edges, axis=-2)
    new_source_vertices = tf.tensordot(source_edges_vertices, source_weight, axes = [[-1], [0]])
    new_destination_vertices = tf.tensordot(destination_edges_vertices, destination_weight, axes = [[-1], [0]])
    new_universe = tf.tensordot(universe, universe_weight, axes=[[-1], [0]])
    new_universe = tf.reshape(new_universe, shape=[-1, 1, new_universe.shape[-1].value)
    return activation(new_edges + new_source_vertices + new_destination_vertices + new_universe + edge_bias)

def aggregate_edges_of_vertex(name, units, vertex_edge_matrix, edges)
    edges_weight = create_variable(name + "_vertex", shape=[edges.shape[-1].value, units])
    vertex = tf.tensordot(edges, vertex_edge_matrix, axes = [[-2], [-1]])
    vertex = tf.transpose(vertex, perm=[0, 2, 1])
    return tf.tensordot(vertex, edges_weight, axes = [[-1], [0]])

def new_vertex(units, activation, source_vertices, destination_vertices, vertices, edges, universe):
    source_edges = aggregate_edges_of_vertex("source", units, activation, source_vertices, edges)
    destination_edges = aggregate_edges_of_vertex("destination", units, activation, destination_vertices, edges)
    vertex_weight = create_variable("vertex", shape=[vertices.shape[-1].value, units])
    
    destionation_edges_weight = create_variable("source_vertex_edges", shape=[edges.shape[-1].value])
    source_edges = tf.tensordot(edges, source_vertices, axes = [[-2], [-1]])
    source_edges = tf.transpose(source_edges, perm=[0, 2, 1])
    destination_edges = tf.tensordot(edges, destination_vertices, axes = [[-2], [-1]])
    destination_edges = tf.transpose(destination_vertices, perm=[0, 2, 1])



def gn_block(source_vertices, destination_vertices, vertices, source_edges, destination_edges, edges_attribute, universe):
    """
    source edges: [vertices_count, edge_count]
    source edges: [vertices_count, edge_count]
    vertices shape: [batch_size, vertices_count, vertex_attributes]
    source_edges shape: [edges_count]
    destination_edges shape: [edges_count]
    edges_attribute shape: [batch_size, edges_count, edge_attributes]
    universe shape: [batch_size, universe_attributes]
    """
    new_edge_edge = {kkk

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

