import tensorflow as tf
import Model


class Xception(Model.Model):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(Xception, self).__init__(args, dtype, label_dim, initializer)
        self.dropout_rate = args.dropout_rate
        self.image_height = 299
        self.image_width = 299
        self.channel_multiplier = 1

    def conv_bn_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def conv_bn_relu_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        conv_bn = self.conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, padding=padding, training=training, scope_name=scope_name)
        relu = tf.nn.relu(conv_bn, name=scope_name + '_conv_bn_relu')
        return relu

    def separable_conv_bn_layer(self, input, depthwise_filter_shape, pointwise_filter_shape, strides, padding, training, scope_name):
        depthwise_filter = tf.get_variable(shape=depthwise_filter_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_depthwise_filter')
        pointwise_filter = tf.get_variable(shape=pointwise_filter_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_pointwise_filter')
        separable_conv = tf.nn.separable_conv2d(input, depthwise_filter=depthwise_filter, pointwise_filter=pointwise_filter, strides=strides, padding=padding, name=scope_name + '_separable_conv')
        bn = tf.layers.batch_normalization(separable_conv, training=training, name=scope_name + '_separable_conv_bn')
        return bn

    def relu_separable_conv_bn_layer(self, input, depthwise_filter_shape, pointwise_filter_shape, strides, padding, training, scope_name):
        relu = tf.nn.relu(input, name=scope_name + '_relu')
        conv_bn = self.separable_conv_bn_layer(relu, depthwise_filter_shape=depthwise_filter_shape, pointwise_filter_shape=pointwise_filter_shape, strides=strides, padding=padding, training=training, scope_name=scope_name + '_relu_conv_bn')
        return conv_bn

    def separable_conv_bn_relu_layer(self, input, depthwise_filter_shape, pointwise_filter_shape, strides, padding, training, scope_name):
        conv_bn = self.separable_conv_bn_layer(input, depthwise_filter_shape=depthwise_filter_shape, pointwise_filter_shape=pointwise_filter_shape, strides=strides, padding=padding, training=training, scope_name=scope_name + '_conv_bn')
        relu = tf.nn.relu(conv_bn, name=scope_name + '_conv_bn_relu')
        return relu

    def model(self, images, training):
        with tf.variable_scope('entry_flow') as scope:
            conv1 = self.conv_bn_relu_layer(images, kernel_shape=[3, 3, 3, 32], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope.name + '_conv1')
            conv2 = self.conv_bn_relu_layer(conv1, kernel_shape=[3, 3, 32, 64], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope.name + '_conv2')

            res1 = self.conv_bn_layer(conv2, kernel_shape=[1, 1, 64, 128], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope.name + '_res1')
            separable_conv1 = self.separable_conv_bn_layer(conv2, depthwise_filter_shape=[3, 3, 64, self.channel_multiplier], pointwise_filter_shape=[1, 1, 64 * self.channel_multiplier, 128], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv1')
            separable_conv2 = self.relu_separable_conv_bn_layer(separable_conv1, depthwise_filter_shape=[3, 3, 128, self.channel_multiplier], pointwise_filter_shape=[1, 1, 128 * self.channel_multiplier, 128], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv2')
            max_pool1 = tf.nn.max_pool(separable_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name + 'max_pool1')
            block1 = tf.add(max_pool1, res1, name=scope.name + '_block1')

            res2 = self.conv_bn_layer(block1, kernel_shape=[1, 1, 128, 256], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope.name + '_res2')
            separable_conv3 = self.relu_separable_conv_bn_layer(block1, depthwise_filter_shape=[3, 3, 128, self.channel_multiplier], pointwise_filter_shape=[1, 1, 128 * self.channel_multiplier, 256], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv3')
            separable_conv4 = self.relu_separable_conv_bn_layer(separable_conv3, depthwise_filter_shape=[3, 3, 256, self.channel_multiplier], pointwise_filter_shape=[1, 1, 256 * self.channel_multiplier, 256], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv4')
            max_pool2 = tf.nn.max_pool(separable_conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name + 'max_pool2')
            block2 = tf.add(max_pool2, res2, name=scope.name + '_block2')

            res3 = self.conv_bn_layer(block2, kernel_shape=[1, 1, 256, 728], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope.name + '_res3')
            separable_conv5 = self.relu_separable_conv_bn_layer(block2, depthwise_filter_shape=[3, 3, 256, self.channel_multiplier], pointwise_filter_shape=[1, 1, 256 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv5')
            separable_conv6 = self.relu_separable_conv_bn_layer(separable_conv5, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv6')
            max_pool3 = tf.nn.max_pool(separable_conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name + 'max_pool3')
            net = tf.add(max_pool3, res3, name=scope.name + '_block3')

        with tf.variable_scope('middle_flow') as scope:
            for i in range(7, 15):  # repeated 8 times
                block_net = self.relu_separable_conv_bn_layer(net, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv' + str(7 + (i - 7) * 3))
                block_net = self.relu_separable_conv_bn_layer(block_net, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv' + str(8 + (i - 7) * 3))
                block_net = self.relu_separable_conv_bn_layer(block_net, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv' + str(9 + (i - 7) * 3))
                net = tf.add(block_net, net, name=scope.name + '_block' + str(i - 3))

        with tf.variable_scope('exit_flow') as scope:
            res4 = self.conv_bn_layer(net, kernel_shape=[1, 1, 728, 1024], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope.name + '_res4')
            separable_conv31 = self.relu_separable_conv_bn_layer(net, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 728], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv31')
            separable_conv32 = self.relu_separable_conv_bn_layer(separable_conv31, depthwise_filter_shape=[3, 3, 728, self.channel_multiplier], pointwise_filter_shape=[1, 1, 728 * self.channel_multiplier, 1024], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv32')
            max_pool4 = tf.nn.max_pool(separable_conv32, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name + '_max_pool4')
            block5 = tf.add(max_pool4, res4, name=scope.name + '_block12')

            separable_conv12 = self.separable_conv_bn_relu_layer(block5, depthwise_filter_shape=[3, 3, 1024, self.channel_multiplier], pointwise_filter_shape=[1, 1, 1024 * self.channel_multiplier, 1536], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv12')
            separable_conv13 = self.separable_conv_bn_relu_layer(separable_conv12, depthwise_filter_shape=[3, 3, 1536, self.channel_multiplier], pointwise_filter_shape=[1, 1, 1536 * self.channel_multiplier, 2048], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name + '_separable_conv13')

            avg_pool = tf.nn.avg_pool(separable_conv13, ksize=[1, 10, 10, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
            w = tf.get_variable(shape=[2048, self.label_dim], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name='weight')
            b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
            z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

            return z
