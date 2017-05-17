import tensorflow as tf

def scalar_summary(name, tensor):
    #hack for lab pcs:
    if tf.__version__.startswith('0.10'):
        tf.scalar_summary(name, tensor)
    else:
        tf.summary.scalar(name, tensor)

def merge_all_summaries():
    if tf.__version__.startswith('0.10'):
        return tf.merge_all_summaries()
    else:
        return tf.summary.merge_all()

def create_summary_writer(logdir, graph):
    if tf.__version__.startswith('0.10'):
        return tf.train.SummaryWriter(logdir, graph)
    else:
        return tf.summary.FileWriter(logdir, graph)

def initialize_all_variables():
    if tf.__version__.startswith('0.10'):
        return tf.initialize_all_variables()
    else:
        return tf.global_variables_initializer()

def add_variable_summary(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        scalar_summary('mean/'+name, mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        scalar_summary('stddev/' + name, stddev)




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, step):
    return tf.nn.conv2d(x,W, strides=[1,step,step,1], padding="SAME")

def create_conv_layer(h, W, b, step, layer_name=None):
    W_conv = weight_variable(W)
    b_conv = bias_variable(b)

    if layer_name is not None:
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                add_variable_summary(W_conv, layer_name+"/weights")
            with tf.name_scope("biases"):
                add_variable_summary(b_conv, layer_name+"/biases")

    return tf.nn.relu(conv2d(h,W_conv,step) + b_conv), W_conv, b_conv


def create_fc_layer(h, W, b, activation_function = None, layer_name = None):
    W_fc = weight_variable(W)
    b_fc = bias_variable(b)

    if layer_name is not None:
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                add_variable_summary(W_fc, layer_name+"/weights")
            with tf.name_scope("biases"):
                add_variable_summary(b_fc, layer_name+"/biases")

    #no activation function => linear layer
    if activation_function is None:
        return tf.matmul(h,W_fc) + b_fc, W_fc, b_fc

    return activation_function(tf.matmul(h,W_fc) + b_fc), W_fc, b_fc


def create_fc_layer_relu(h, W, b, layer_name = None):
    W_fc = weight_variable(W)
    b_fc = bias_variable(b)

    if layer_name is not None:
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                add_variable_summary(W_fc, layer_name+"/weights")
            with tf.name_scope("biases"):
                add_variable_summary(b_fc, layer_name+"/biases")

    return tf.nn.relu(tf.matmul(h, W_fc) + b_fc)

def create_fc_layer_linear(h, W, b, layer_name = None):
    W_fc = weight_variable(W)
    b_fc = bias_variable(b)

    if layer_name is not None:
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                add_variable_summary(W_fc, layer_name+"/weights")
            with tf.name_scope("biases"):
                add_variable_summary(b_fc, layer_name+"/biases")

    return (tf.matmul(h,W_fc) + b_fc), W_fc, b_fc

def create_fc_layer_tanh(h,W,b, layer_name):
    W_fc = weight_variable(W)
    b_fc = bias_variable(b)

    if layer_name is not None:
        with tf.name_scope(layer_name):
            with tf.name_scope("weights"):
                add_variable_summary(W_fc, layer_name+"/weights")
            with tf.name_scope("biases"):
                add_variable_summary(b_fc, layer_name+"/biases")


    return tf.nn.tanh(tf.matmul(h, W_fc) + b_fc)

def create_fc_layer_sigmoid(h,W,b):
    W_fc = weight_variable(W)
    b_fc = bias_variable(b)

    return tf.nn.sigmoid(tf.matmul(h, W_fc) + b_fc)

def clipped_error(x):
    # Huber loss
    try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
