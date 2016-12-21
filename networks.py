# -*- coding: utf-8 -*-
#
#
import tensorflow as tf
import tensorlayer as tl


# ------------------------------------------------------- #
def __vgg16_conv_layers(net_in):
    # with tf.name_scope('preprocess'):
    #     """
    #     Notice that we include a preprocessing layer that takes the RGB image
    #     with pixels values in the range of 0-255 and subtracts the mean image
    #     values (calculated over the entire ImageNet training set).
    #     """
    #     mean=tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    #     net_in.outputs=net_in.outputs - mean
    """ conv1 """
    network = tl.layers.Conv2dLayer(net_in,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 3, 64],  # 64 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 64, 64],  # 64 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv1_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool1')
    """ conv2 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 64, 128],  # 128 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 128, 128],  # 128 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv2_2')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool2')
    """ conv3 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 128, 256],  # 256 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 256, 256],  # 256 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv3_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool3')
    """ conv4 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 256, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv4_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool4')
    """ conv5 """
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_1')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_2')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[3, 3, 512, 512],  # 512 features for each 3x3 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='conv5_3')
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool5')
    return network


def __vgg16_fc_layers(net):
    network = tl.layers.FlattenLayer(net, name='flatten')
    network = tl.layers.DenseLayer(network, n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc1_relu')
    network = tl.layers.DenseLayer(network, n_units=4096,
                                   act=tf.nn.relu,
                                   name='fc2_relu')
    network = tl.layers.DenseLayer(network, n_units=1000,
                                   act=tf.identity,
                                   name='fc3_relu')
    return network


def vgg16(net_in, reuse=False, drop_fc_layers=False):
    """
    Build the VGG 16 Model.
    Parameters
    -----------
    net_in : input layer of vgg16 with shape [batch, height, width, 3] values scaled [0, 255.]

    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = __vgg16_conv_layers(net_in)
        if not drop_fc_layers:
            network = __vgg16_fc_layers(network)
    return network


# ------------------------------------------------------- #
def vgg19(net_in, reuse=False, drop_fc_layers=False):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    net_in : input layer of vgg19 with shape [batch, height, width, 3] values scaled [0, 255.]
    """
    """ conv1 """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = tl.layers.Conv2dLayer(net_in,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 3, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv1_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 64],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv1_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool1')
        """ conv2 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 64, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv2_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 128],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv2_2')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool2')
        """ conv3 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 128, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_3')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 256],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv3_4')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool3')
        """ conv4 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 256, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_3')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv4_4')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool4')
        """ conv5 """
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_1')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_2')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_3')
        network = tl.layers.Conv2dLayer(network,
                                        act=tf.nn.relu,
                                        shape=[3, 3, 512, 512],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='conv5_4')
        network = tl.layers.PoolLayer(network,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      pool=tf.nn.max_pool,
                                      name='pool5')
        if not drop_fc_layers:
            """ fc 6~8 """
            network = tl.layers.FlattenLayer(network, name='flatten')
            network = tl.layers.DenseLayer(network, n_units=4096,
                                           act=tf.nn.relu, name='fc6')
            network = tl.layers.DenseLayer(network, n_units=4096,
                                           act=tf.nn.relu, name='fc7')
            network = tl.layers.DenseLayer(network, n_units=1000,
                                           act=tf.identity, name='fc8')

    return network
