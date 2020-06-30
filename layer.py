from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow as tf
import tensorflow.contrib.slim as slim

def concat2d(x1,x2):
    """ concatenation without offset check"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    try:
        tf.equal(x1_shape[0:-2], x2_shape[0: -2])
    except:
        print("x1_shape: %s"%str(x1.get_shape().as_list()))
        print("x2_shape: %s"%str(x2.get_shape().as_list()))
        raise ValueError("Cannot concatenate tensors with different shape, igonoring feature map depth")
    return tf.concat([x1, x2], 3)

def normalize(inp, activation, reuse=tf.AUTO_REUSE, scope='', form='batch_norm', is_training=True):
    if form == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope, is_training=is_training)
    elif form == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif form == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

def conv_block(inp, cweight, bweight, scope='', bn=True, is_training=True):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    conv = tf.nn.conv2d(inp, cweight, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bweight)

    if bn == True:
        normed = normalize(conv, tf.nn.relu, scope=scope, is_training=is_training)
        return normed
    else:
        return conv
    # relu = tf.nn.leaky_relu(normed)
    # normalize = batch_norm(relu, True)


def deconv_block(inp, cweight, bweight, scope='', is_training=True):
    # x_shape = tf.shape(inp)
    x_shape = inp.get_shape().as_list()
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    deconv = tf.nn.conv2d_transpose(inp, cweight, output_shape, strides=[1,2,2,1], padding='SAME')
    deconv = tf.nn.bias_add(deconv, bweight)

    normed = normalize(deconv, tf.nn.relu, scope=scope, is_training=is_training)
    return normed


def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding)

def lrn(x, radius, alpha, beta, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def fc(x, wweight, bweight, activation=None):
    """Create a fully connected layer."""
    
    act = tf.nn.xw_plus_b(x, wweight, bweight)

    if activation is 'relu':
        return tf.nn.relu(act)
    elif activation is 'leaky_relu':
        return tf.nn.leaky_relu(act)
    elif activation is None:
        return act
    else:
        raise NotImplementedError

