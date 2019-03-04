import tensorflow as tf
import Model
import math


class MobileNet(Model.Model):
    def __init__(self, args, dtype, label_dim, initializer=None):
        super(MobileNet, self).__init__(args, dtype, label_dim, initializer)
        self.initializer = initializer if initializer is not None else tf.initializers.zeros()
        self.channel_multiplier = 1

    def conv_bn_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def conv_bn_relu6_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        conv_bn = self.conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, padding=padding, training=training, scope_name=scope_name)
        relu6 = tf.nn.relu6(conv_bn, name=scope_name + '_conv_bn_relu6')
        return relu6

    def depthwise_conv_bn_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        depthwise_conv = tf.nn.depthwise_conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_depthwise_conv')
        depthwise_conv_bn = tf.layers.batch_normalization(depthwise_conv, training=training, name=scope_name + '_depthwise_conv_bn')
        return depthwise_conv_bn

    def depthwise_conv_bn_relu6_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        depthwise_conv_bn = self.depthwise_conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, padding=padding, training=training, scope_name=scope_name)
        depthwise_conv_bn_relu6 = tf.nn.relu6(depthwise_conv_bn, name=scope_name + '_depthwise_conv_bn_relu6')
        return depthwise_conv_bn_relu6

    def separable_conv_bn_layer(self, input, depthwise_filter_shape, pointwise_filter_shape, strides, padding, training, scope_name):
        depthwise_filter = tf.get_variable(shape=depthwise_filter_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_depthwise_filter')
        pointwise_filter = tf.get_variable(shape=pointwise_filter_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_pointwise_filter')
        separable_conv = tf.nn.separable_conv2d(input, depthwise_filter=depthwise_filter, pointwise_filter=pointwise_filter, strides=strides, padding=padding, name=scope_name + '_separable_conv')
        bn = tf.layers.batch_normalization(separable_conv, training=training, name=scope_name + '_separable_conv_bn')
        return bn

    def separable_conv_bn_relu6_layer(self, input, depthwise_filter_shape, pointwise_filter_shape, strides, padding, training, scope_name):
        separable_conv_bn = self.separable_conv_bn_layer(input, depthwise_filter_shape=depthwise_filter_shape, pointwise_filter_shape=pointwise_filter_shape, strides=strides, padding=padding, training=training, scope_name=scope_name)
        separable_conv_bn_relu6 = tf.nn.relu6(separable_conv_bn, name=scope_name + '_separable_conv_bn_relu6')
        return separable_conv_bn_relu6


class MobileNet_V1(MobileNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(MobileNet_V1, self).__init__(args, dtype, label_dim, initializer)
        self.width_multiple = 1
        self.channel_num = [int(math.ceil(32 * self.width_multiple)), int(math.ceil(64 * self.width_multiple)), int(math.ceil(128 * self.width_multiple)), int(math.ceil(256 * self.width_multiple)), int(math.ceil(512 * self.width_multiple)), int(math.ceil(1024 * self.width_multiple))]
        self.resolution = 224
        self.image_height = self.resolution
        self.image_width = self.resolution

    def model(self, images, training):
        conv1 = self.conv_bn_relu6_layer(images, kernel_shape=[3, 3, 3, self.channel_num[0]], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv1")
        conv2 = self.separable_conv_bn_relu6_layer(conv1, depthwise_filter_shape=[3, 3, self.channel_num[0], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[0] * self.channel_multiplier, self.channel_num[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv2")
        conv3 = self.separable_conv_bn_relu6_layer(conv2, depthwise_filter_shape=[3, 3, self.channel_num[1], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[1] * self.channel_multiplier, self.channel_num[2]], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv3")
        conv4 = self.separable_conv_bn_relu6_layer(conv3, depthwise_filter_shape=[3, 3, self.channel_num[2], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[2] * self.channel_multiplier, self.channel_num[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv4")
        conv5 = self.separable_conv_bn_relu6_layer(conv4, depthwise_filter_shape=[3, 3, self.channel_num[2], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[2] * self.channel_multiplier, self.channel_num[3]], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv5")
        conv6 = self.separable_conv_bn_relu6_layer(conv5, depthwise_filter_shape=[3, 3, self.channel_num[3], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[3] * self.channel_multiplier, self.channel_num[3]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv6")
        conv7 = self.separable_conv_bn_relu6_layer(conv6, depthwise_filter_shape=[3, 3, self.channel_num[3], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[3] * self.channel_multiplier, self.channel_num[4]], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv7")
        conv8 = self.separable_conv_bn_relu6_layer(conv7, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv8")
        conv9 = self.separable_conv_bn_relu6_layer(conv8, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv9")
        conv10 = self.separable_conv_bn_relu6_layer(conv9, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv10")
        conv11 = self.separable_conv_bn_relu6_layer(conv10, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv11")
        conv12 = self.separable_conv_bn_relu6_layer(conv11, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv12")
        conv13 = self.separable_conv_bn_relu6_layer(conv12, depthwise_filter_shape=[3, 3, self.channel_num[4], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[4] * self.channel_multiplier, self.channel_num[5]], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv13")
        conv14 = self.separable_conv_bn_relu6_layer(conv13, depthwise_filter_shape=[3, 3, self.channel_num[5], self.channel_multiplier], pointwise_filter_shape=[1, 1, self.channel_num[5] * self.channel_multiplier, self.channel_num[5]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv14")

        avg_pool = tf.nn.avg_pool(conv14, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.channel_num[5], self.label_dim], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class MobileNet_V2(MobileNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(MobileNet_V2, self).__init__(args, dtype, label_dim, initializer)
        self.width_multiple = 1
        self.channel_num = [int(math.ceil(16 * self.width_multiple)), int(math.ceil(24 * self.width_multiple)), int(math.ceil(32 * self.width_multiple)), int(math.ceil(64 * self.width_multiple)), int(math.ceil(96 * self.width_multiple)), int(math.ceil(160 * self.width_multiple)), int(math.ceil(320 * self.width_multiple)), int(math.ceil(1280 * self.width_multiple))]
        self.resolution = 224
        self.image_height = self.resolution
        self.image_width = self.resolution

    def bottleneck_block(self, input, expansion_factor, out_channels, strides, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        res = self.conv_bn_relu6_layer(input, kernel_shape=[1, 1, in_channels, out_channels], strides=strides, padding="SAME", training=training, scope_name=scope_name + '_res')
        conv1 = self.conv_bn_relu6_layer(input, kernel_shape=[1, 1, in_channels, in_channels * expansion_factor], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_conv1')
        conv2 = self.depthwise_conv_bn_relu6_layer(conv1, kernel_shape=[3, 3, in_channels * expansion_factor, self.channel_multiplier], strides=strides, padding="SAME", training=training, scope_name=scope_name + '_conv2')
        conv3 = self.conv_bn_layer(conv2, kernel_shape=[1, 1, in_channels * expansion_factor * self.channel_multiplier, out_channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_conv3')
        out = tf.add(conv3, res, name=scope_name + '_out')
        return out

    def model(self, images, training):
        conv1 = self.conv_bn_relu6_layer(images, kernel_shape=[3, 3, 3, 32], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name="conv1")
        conv2 = self.bottleneck_block(conv1, expansion_factor=1, out_channels=self.channel_num[0], strides=[1, 1, 1, 1], training=training, scope_name='conv2')
        conv3 = self.bottleneck_block(conv2, expansion_factor=6, out_channels=self.channel_num[1], strides=[1, 2, 2, 1], training=training, scope_name='conv3')
        conv4 = self.bottleneck_block(conv3, expansion_factor=6, out_channels=self.channel_num[1], strides=[1, 1, 1, 1], training=training, scope_name='conv4')
        conv5 = self.bottleneck_block(conv4, expansion_factor=6, out_channels=self.channel_num[2], strides=[1, 2, 2, 1], training=training, scope_name='conv5')
        conv6 = self.bottleneck_block(conv5, expansion_factor=6, out_channels=self.channel_num[2], strides=[1, 1, 1, 1], training=training, scope_name='conv6')
        conv7 = self.bottleneck_block(conv6, expansion_factor=6, out_channels=self.channel_num[2], strides=[1, 1, 1, 1], training=training, scope_name='conv7')
        conv8 = self.bottleneck_block(conv7, expansion_factor=6, out_channels=self.channel_num[3], strides=[1, 2, 2, 1], training=training, scope_name='conv8')
        conv9 = self.bottleneck_block(conv8, expansion_factor=6, out_channels=self.channel_num[3], strides=[1, 1, 1, 1], training=training, scope_name='conv9')
        conv10 = self.bottleneck_block(conv9, expansion_factor=6, out_channels=self.channel_num[3], strides=[1, 1, 1, 1], training=training, scope_name='conv10')
        conv11 = self.bottleneck_block(conv10, expansion_factor=6, out_channels=self.channel_num[3], strides=[1, 1, 1, 1], training=training, scope_name='conv11')
        conv12 = self.bottleneck_block(conv11, expansion_factor=6, out_channels=self.channel_num[4], strides=[1, 1, 1, 1], training=training, scope_name='conv12')
        conv13 = self.bottleneck_block(conv12, expansion_factor=6, out_channels=self.channel_num[4], strides=[1, 1, 1, 1], training=training, scope_name='conv13')
        conv14 = self.bottleneck_block(conv13, expansion_factor=6, out_channels=self.channel_num[4], strides=[1, 1, 1, 1], training=training, scope_name='conv14')
        conv15 = self.bottleneck_block(conv14, expansion_factor=6, out_channels=self.channel_num[5], strides=[1, 2, 2, 1], training=training, scope_name='conv15')
        conv16 = self.bottleneck_block(conv15, expansion_factor=6, out_channels=self.channel_num[5], strides=[1, 1, 1, 1], training=training, scope_name='conv16')
        conv17 = self.bottleneck_block(conv16, expansion_factor=6, out_channels=self.channel_num[5], strides=[1, 1, 1, 1], training=training, scope_name='conv17')
        conv18 = self.bottleneck_block(conv17, expansion_factor=6, out_channels=self.channel_num[6], strides=[1, 1, 1, 1], training=training, scope_name='conv18')
        conv19 = self.conv_bn_relu6_layer(conv18, kernel_shape=[1, 1, self.channel_num[6], self.channel_num[7]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name="conv19")

        avg_pool = tf.nn.avg_pool(conv19, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.channel_num[7], self.label_dim], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z
