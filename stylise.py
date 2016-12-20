# -*- coding: utf-8 -*-
#

import os
import sys
from operator import itemgetter

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from scipy.misc import imread, imresize, imsave

from networks import vgg16, vgg19


# define paramters.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('base_img_path', 'img/normal.jpg', 'path of base image.')
flags.DEFINE_string('style_img_path', 'img/starry_night.jpg', 'path of style image.')
flags.DEFINE_string('out_img_path', 'img/after.jpg', 'path of out image.')
flags.DEFINE_integer('img_height', 224, 'height of input image.')
flags.DEFINE_integer('img_width', 224, 'width of input image.')
flags.DEFINE_integer('channels', 3, 'channels of input image.')
flags.DEFINE_float('content_weight', 0.005, 'weight of base loss.')
flags.DEFINE_float('style_weight', 1., 'weight of style loss.')
flags.DEFINE_float('variation_weight', 1., 'weight of variational loss.')
# flags.DEFINE_string('params_path', 'vgg16_weights.npz', 'weights parameters of pre-trained vgg16 model.')
flags.DEFINE_integer('n_epochs', 10, 'iteration epochs.')
flags.DEFINE_integer('n_steps', 200, 'iteration steps with each epoch.')
flags.DEFINE_float('learning_rate', 10., 'learning rate of training.')

VGG_MEAN = [103.939, 116.779, 123.68]
MODELS = {'vgg16': vgg16, 'vgg19': vgg19}
WEIGHTS_PATH = {
    'vgg16': 'weights/vgg16_weights.npz',
    'vgg19': 'weights/vgg19_weights.npz'
}

assert FLAGS.channels == 3


class StyleTransformer():
    def __init__(self, network_name='vgg16', resize=False):
        # build network
        self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.9))))

        self._content_layer_name = 'conv4_2'
        # self._style_layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self._style_layer_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        # build network
        imgs = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.channels])
        net_in = tl.layers.InputLayer(imgs, name='input_layer')
        self._network = MODELS[network_name](net_in, drop_fc_layers=True)
        # initialize and assign vgg params.
        self._sess.run(tf.variables_initializer(self._network.all_params))
        params = np.load(WEIGHTS_PATH[network_name], encoding='latin1')
        params = params.items()
        params.sort(key=itemgetter(0))
        params = [v for k, v in params if k.startswith('conv')]
        tl.files.assign_params(self._sess, params, self._network)
        all_layers = dict((layer.name.split('/')[0], layer) for layer in self._network.all_layers)

        # get content features.
        base_arr = self._preprocess_img(FLAGS.base_img_path, resize=resize)
        content_layer = self._sess.run(all_layers[self._content_layer_name], feed_dict={imgs: base_arr})
        self._content_features = tf.constant(content_layer[0, :, :, :], dtype=tf.float32)
        del params

        # get style features
        style_arr = self._preprocess_img(FLAGS.style_img_path, resize=resize)
        self._style_features = dict()
        for layer_name in self._style_layer_names:
            style_layer = self._sess.run(all_layers[layer_name], feed_dict={imgs: style_arr})
            self._style_features[layer_name] = tf.constant(style_layer[0, :, :, :], dtype=tf.float32)

        # get out network
        _, height, width, _ = base_arr.shape
        initial_value = np.random.uniform(0, 255., size=(1, height, width, FLAGS.channels))
        initial_value -= np.array(VGG_MEAN).reshape(1, 1, 1, 3)
        self._out_tensor = tf.Variable(initial_value, name='out', dtype=tf.float32)
        net_in_ = tl.layers.InputLayer(tf.concat(concat_dim=0, values=[self._out_tensor]), name='out_input_layer')
        self._network_ = MODELS[network_name](net_in_, reuse=True, drop_fc_layers=True)
        self._sess.run(tf.variables_initializer([self._out_tensor]))

    # preprocess image.
    def _preprocess_img(self, img_path, resize=False):
        x = imread(img_path, mode='RGB')
        if resize:
            x = imresize(x, size=(FLAGS.img_height, FLAGS.img_width, FLAGS.channels))
        x = x.astype(np.float32)
        # remove mean.
        x -= np.array(VGG_MEAN).reshape(1, 1, 3)
        x = np.expand_dims(x, axis=0)
        return x

    def _restore_img(self, x):
        # x = x.reshape(FLAGS.img_height, FLAGS.img_width, FLAGS.channels)
        x += np.array(VGG_MEAN).reshape(1, 1, 3)
        return np.clip(x, 0, 255).astype('uint8')

    # the gram matrix of an image tensor (feature-wise outer product)
    def _gram_matrix(self, x):
        assert len(x.get_shape()._dims) == 3
        x = tf.transpose(x, perm=[2, 0, 1])
        shape = [s.value for s in x.get_shape()]
        features = tf.reshape(x, [-1, np.prod(shape[1:])])
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    # the "style loss" is designed to maintain
    # the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of
    # feature maps from the style reference image
    # and from the generated image
    def _style_loss(self, style, out):
        assert len(style.get_shape()._dims) == 3
        assert len(out.get_shape()._dims) == 3
        S = self._gram_matrix(style)
        C = self._gram_matrix(out)
        img_nrows, img_ncols, channels = [x.value for x in style.get_shape()]
        # img_nrows, img_ncols, channels = FLAGS.img_height, FLAGS.img_width, FLAGS.channels
        size = img_nrows * img_ncols
        return tf.nn.l2_loss(S-C) / (2. * (channels ** 2) * (size ** 2))

    # an auxiliary loss function
    # designed to maintain the "content" of the
    # base image in the generated image
    def _content_loss(self, base, out):
        return tf.nn.l2_loss(base-out)

    # the 3rd loss function, total variation loss,
    # designed to keep the generated image locally coherent
    def _variation_loss(self, x):
        assert len(x.get_shape()._dims) == 3
        height, width, _ = [s.value for s in x.get_shape()]
        height_diff = x[:height-1, :width-1, :] - x[1:, :width-1, :]
        width_diff = x[:height-1, :width-1, :] - x[:height-1, 1:, :]
        return tf.nn.l2_loss(height_diff) + tf.nn.l2_loss(width_diff)

    def _loss(self):
        # compute total loss
        layers_ = dict((layer.name.split('/')[0], layer) for layer in self._network_.all_layers)
        content_features_ = layers_[self._content_layer_name+str('_1')][0, :, :, :]
        loss = self._content_loss(self._content_features, content_features_) * FLAGS.content_weight
        for layer_name in self._style_layer_names:
            style_features_ = layers_[layer_name+str('_1')][0, :, :, :]
            loss += self._style_loss(self._style_features[layer_name], style_features_) \
                                                    * FLAGS.style_weight / len(self._style_layer_names)
        loss += self._variation_loss(self._out_tensor[0, :, :, :]) * FLAGS.variation_weight
        return loss

    def train(self):
        # define training ops
        loss = self._loss()
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(loss, var_list=[self._out_tensor])
        # initialize all uninitialized parameters.
        uninitialized_vars = \
            [var for var in tf.global_variables() if not tf.is_variable_initialized(var).eval(session=self._sess)]
        self._sess.run(tf.variables_initializer(uninitialized_vars))
        # training.
        for n_epoch in range(FLAGS.n_epochs):
            for _ in range(FLAGS.n_steps):
                _, ls = self._sess.run([train_op, loss])
            print('loss:', ls)
            print('saving image at epoch {}'.format(n_epoch))
            out_img = self._sess.run(self._out_tensor[0, :, :, :])
            out_img = self._restore_img(out_img)
            imsave('img/after_{}.jpg'.format(n_epoch), out_img)


if __name__ == '__main__':
    style_transformer = StyleTransformer('vgg16', resize=True)
    style_transformer.train()
