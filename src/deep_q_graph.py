import tensorflow as tf
import numpy as np

import utils as ut

class NetworkArchitecture(object):
    def __init__(self):
        self.weights = []
        self.biases = []

    def copy_weights(self, other_graph):
        for weight, other_weight in zip(self.weights, other_graph.weights):
            weight.assign(other_weight).op.run()

        for bias, other_bias in zip(self.biases, other_graph.biases):
            bias.assign(other_bias).op.run()


class DeepQArchitecture(NetworkArchitecture):
    def __init__(self, history_length, num_actions, input, name = None):
        super(DeepQArchitecture, self).__init__()


        #create the convolutional layer
        if name is not None:
            h_conv1, W_conv1, b_conv1 = ut.create_conv_layer(input, [8, 8, history_length, 32], [32], 4, name+"/conv1")
            h_conv2, W_conv2, b_conv2 = ut.create_conv_layer(h_conv1, [4, 4, 32, 64], [64], 2, name+"/conv2")
            h_conv3, W_conv3, b_conv3 = ut.create_conv_layer(h_conv2, [3, 3, 64, 64], [64], 1, name+"/conv3")
        else:
            h_conv1, W_conv1, b_conv1 = ut.create_conv_layer(input, [8, 8, history_length, 32], [32], 4)
            h_conv2, W_conv2, b_conv2 = ut.create_conv_layer(h_conv1, [4, 4, 32, 64], [64], 2)
            h_conv3, W_conv3, b_conv3 = ut.create_conv_layer(h_conv2, [3, 3, 64, 64], [64], 1)

        self.weights.extend([W_conv1, W_conv2, W_conv3])
        self.biases.extend([b_conv1, b_conv2, b_conv3])
        
        h_shape = h_conv3.get_shape()
        h_shape_eval = h_shape[1] * h_shape[2] * h_shape[3]
        
        h_count = int(h_shape_eval.value)

        h_reshape = tf.reshape(h_conv3, [-1, h_count])

        #create fully connected layer
        if name is not None:
            h_fc1, W_fc1, b_fc1 = ut.create_fc_layer(h_reshape, [h_count, 512], [512], tf.nn.relu, name+"/fc1")
            self.q_values, W_fc2, b_fc2 = ut.create_fc_layer_linear(h_fc1, [512,num_actions], [num_actions], name+"/fc2")
        else:
            h_fc1, W_fc1, b_fc1 = ut.create_fc_layer(h_reshape, [h_count, 512], [512], tf.nn.relu)
            self.q_values, W_fc2, b_fc2 = ut.create_fc_layer_linear(h_fc1, [512,num_actions], [num_actions])

        self.weights.extend([W_fc1, W_fc2])
        self.biases.extend([b_fc1, b_fc2])


class SimpleDeepQArchitecture(DeepQArchitecture):
    def __init__(self, history_length, num_actions, input, name = ""):
        super(DeepQArchitecture, self).__init__()

        #set up a simpler version of the network with only one conv layer
        h_conv1, W_conv1, b_conv1 = ut.create_conv_layer(input, [8.8, history_length, 32], [32], 4, name+"/conv1")

        h_reshape = tf.reshape(h_conv1, [-1, (16*16*64)])

        h_fc1, W_fc1, b_fc1 = ut.create_fc_layer(h_reshape, [16*16*64], [512], tf.nn.relu, name+"/fc1")
        self.q_values, W_fc2, b_fc2 = ut.create_fc_layer_linear(h_fc1, [512, num_actions], [num_actions], name+"/fc1")

        self.weights.extend([W_conv1, W_fc1, W_fc2])
        self.biases.extend([b_conv1, b_fc1, b_fc2])

