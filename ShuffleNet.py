import tensorflow as tf
import Model
import math


class ShuffleNet(Model.Model):
    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ShuffleNet, self).__init__(args, dtype, label_dim, initializer)
        self.initializer = initializer if initializer is not None else tf.initializers.zeros()
        self.groups_list = [1, 2, 3, 4, 8]
        self.out_channels_list = [[144, 288, 576], [200, 400, 800], [240, 480, 960], [272.544, 1088], [384, 768, 1536]]
        self.channel_ratio = 1
        self.groups = self.groups_list[4]
        self.out_channels = self.out_channels_list[4]
        self.out_channels = [int(math.ceil(self.out_channels[0] * self.channel_ratio)), int(math.ceil(self.out_channels[1] * self.channel_ratio)), int(math.ceil(self.out_channels[2] * self.channel_ratio))]
        self.resolution = 224
        self.image_height = self.resolution
        self.image_width = self.resolution

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

    def group_conv(self, input, kernel_shape, strides, cardinality, out_channels, padding, scope_name):
        in_channels = input.shape.as_list()[-1]
        assert in_channels % cardinality == 0 and out_channels % cardinality == 0 and len(kernel_shape) == 2
        group_channels = in_channels // cardinality
        group_filter_num = out_channels // cardinality

        kernel = tf.get_variable(shape=kernel_shape + [in_channels, group_filter_num], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + 'depthwise_kernel')
        group_conv = tf.nn.depthwise_conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_group_conv')
        group_conv_shape = group_conv.shape.as_list()
        group_conv = tf.reshape(group_conv, group_conv_shape[0:3] + [cardinality, group_channels, group_filter_num])
        group_conv = tf.reduce_sum(group_conv, axis=4)
        group_conv = tf.reshape(group_conv, group_conv_shape[0:3] + [out_channels])

        return group_conv

    def group_conv_native(self, input, kernel_shape, strides, cardinality, out_channels, padding, scope_name):
        in_channels = input.shape.as_list()[-1]
        assert in_channels % cardinality == 0 and out_channels % cardinality == 0 and len(kernel_shape) == 2
        group_channels = in_channels // cardinality
        group_filter_num = out_channels // cardinality

        split_features = tf.split(input, num_or_size_splits=cardinality, axis=3, name=scope_name + '_split_features')
        kernel0 = tf.get_variable(shape=kernel_shape + [group_channels, group_filter_num], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel0')
        kernel1 = tf.get_variable(shape=kernel_shape + [group_channels, group_filter_num], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel1')
        net = tf.concat([tf.nn.conv2d(split_features[0], kernel=kernel0, strides=strides, padding=padding), tf.nn.conv2d(split_features[1], kernel=kernel1, strides=strides, padding=padding)], axis=3)
        for i in range(2, cardinality):
            kernel = tf.get_variable(shape=kernel_shape + [group_channels, group_filter_num], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel' + str(i))
            net = tf.concat([net, tf.nn.conv2d(split_features[i], kernel=kernel, strides=strides, padding=padding)], axis=3)

        return net

    def group_conv_bn_layer(self, input, kernel_shape, strides, cardinality, out_channels, padding, training, scope_name):
        conv = self.group_conv(input, kernel_shape=kernel_shape, strides=strides, cardinality=cardinality, out_channels=out_channels, padding=padding, scope_name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def group_conv_bn_relu6_layer(self, input, kernel_shape, strides, cardinality, out_channels, padding, training, scope_name):
        conv_bn = self.group_conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, cardinality=cardinality, out_channels=out_channels, padding=padding, training=training, scope_name=scope_name)
        relu6 = tf.nn.relu6(conv_bn, name=scope_name + '_conv_bn_relu6')
        return relu6

    def group_conv_bn_layer_native(self, input, kernel_shape, strides, cardinality, out_channels, padding, training, scope_name):
        conv = self.group_conv_native(input, kernel_shape=kernel_shape, strides=strides, cardinality=cardinality, out_channels=out_channels, padding=padding, scope_name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def group_conv_bn_relu6_layer_native(self, input, kernel_shape, strides, cardinality, out_channels, padding, training, scope_name):
        conv_bn = self.group_conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, cardinality=cardinality, out_channels=out_channels, padding=padding, training=training, scope_name=scope_name)
        relu6 = tf.nn.relu6(conv_bn, name=scope_name + '_conv_bn_relu6')
        return relu6

    def channel_shuffle(self, input, groups, scope_name):
        in_shape = input.shape.as_list()
        in_channels = in_shape[-1]
        assert in_channels % groups == 0

        net = tf.reshape(input, in_shape[0:3] + [groups, in_channels // groups])
        net = tf.transpose(net, [0, 1, 2, 4, 3])
        net = tf.reshape(net, in_shape[0:3] + [in_channels], name=scope_name + '_shuffle')

        return net

    def model(self, images, training):
        conv = self.conv_bn_relu6_layer(images, kernel_shape=[3, 3, 3, 24], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name='conv')
        max_pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')
        with tf.variable_scope('stage2') as scope:
            net = self.shuffle_unit2(max_pool, groups=self.groups, out_channels=self.out_channels[0], training=training, scope_name=scope.name + '_0', apply_group_conv=False)
            for i in range(1, 4):
                net = self.shuffle_unit1(net, groups=self.groups, training=training, scope_name=scope.name + '_' + str(i))

        with tf.variable_scope('stage3') as scope:
            net = self.shuffle_unit2(net, groups=self.groups, out_channels=self.out_channels[1], training=training, scope_name=scope.name + '_0')
            for i in range(1, 8):
                net = self.shuffle_unit1(net, groups=self.groups, training=training, scope_name=scope.name + '_' + str(i))

        with tf.variable_scope('stage4') as scope:
            net = self.shuffle_unit2(net, groups=self.groups, out_channels=self.out_channels[2], training=training, scope_name=scope.name + '_0')
            for i in range(1, 4):
                net = self.shuffle_unit1(net, groups=self.groups, training=training, scope_name=scope.name + '_' + str(i))

        avg_pool = tf.nn.max_pool(net, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.out_channels[2], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ShuffleNet_V1(ShuffleNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ShuffleNet_V1, self).__init__(args, dtype, label_dim, initializer)

    def shuffle_unit1(self, input, groups, training, scope_name):
        in_channels = input.shape.as_list()[-1]

        conv1 = self.group_conv_bn_relu6_layer(input, kernel_shape=[1, 1], strides=[1, 1, 1, 1], cardinality=groups, out_channels=in_channels // 4, padding="VALID", training=training, scope_name=scope_name + '_conv1')
        shuffle_features = self.channel_shuffle(conv1, groups=groups, scope_name=scope_name)
        conv2 = self.depthwise_conv_bn_layer(shuffle_features, kernel_shape=[3, 3, in_channels // 4, 1], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_conv2')
        conv3 = self.group_conv_bn_layer(conv2, kernel_shape=[1, 1], strides=[1, 1, 1, 1], cardinality=groups, out_channels=in_channels, padding="VALID", training=training, scope_name=scope_name + '_conv3')

        out = tf.nn.relu6(tf.add(conv3, input), name=scope_name + '_out')
        return out

    def shuffle_unit2(self, input, groups, out_channels, training, scope_name, apply_group_conv=True):
        in_channels = input.shape.as_list()[-1]

        if apply_group_conv:
            conv1 = self.group_conv_bn_relu6_layer(input, kernel_shape=[1, 1], strides=[1, 1, 1, 1], cardinality=groups, out_channels=out_channels // 4, padding="VALID", training=training, scope_name=scope_name + '_conv1')
        else:
            conv1 = self.conv_bn_relu6_layer(input, kernel_shape=[1, 1, in_channels, out_channels // 4], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_conv1')
        shuffle_features = self.channel_shuffle(conv1, groups=groups, scope_name=scope_name)
        conv2 = self.depthwise_conv_bn_layer(shuffle_features, kernel_shape=[3, 3, out_channels // 4, 1], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope_name + '_conv2')
        conv3 = self.group_conv_bn_layer(conv2, kernel_shape=[1, 1], strides=[1, 1, 1, 1], cardinality=groups, out_channels=out_channels - in_channels, padding="VALID", training=training, scope_name=scope_name + '_conv3')

        res = tf.nn.avg_pool(input, strides=[1, 2, 2, 1], ksize=[1, 3, 3, 1], padding="SAME", name=scope_name + '_avg_pool')
        out = tf.concat([res, conv3], axis=3, name=scope_name + '_out')
        return out


class ShuffleNet_V2(ShuffleNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ShuffleNet_V2, self).__init__(args, dtype, label_dim, initializer)

    def shuffle_unit1(self, input, groups, training, scope_name):
        branch1, branch2 = tf.split(input, num_or_size_splits=2, axis=3, name=scope_name + '_branch_split')

        branch2_in_channels = branch2.shape.as_list()[-1]
        conv1 = self.conv_bn_relu6_layer(branch2, kernel_shape=[1, 1, branch2_in_channels, branch2_in_channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_conv1')
        conv2 = self.depthwise_conv_bn_layer(conv1, kernel_shape=[3, 3, branch2_in_channels, 1], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_conv2')
        conv3 = self.conv_bn_relu6_layer(conv2, kernel_shape=[1, 1, branch2_in_channels, branch2_in_channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_conv3')

        branch_concat = tf.concat([branch1, conv3], axis=3, name=scope_name + '_branch_concat')
        shuffle_features = self.channel_shuffle(branch_concat, groups=groups, scope_name=scope_name + '_shuffle')
        return shuffle_features

    def shuffle_unit2(self, input, groups, out_channels, training, scope_name, apply_group_conv=True):
        in_channels = input.shape.as_list()[-1]
        channels = out_channels - in_channels

        branch1_conv1 = self.depthwise_conv_bn_layer(input, kernel_shape=[3, 3, in_channels, 1], strides=[1, 2, 2, 1], padding="SAME", training=training, scope_name=scope_name + '_branch1_conv1')
        branch1_conv2 = self.conv_bn_relu6_layer(branch1_conv1, kernel_shape=[1, 1, in_channels, in_channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_branch1_conv2')

        branch2_conv1 = self.conv_bn_relu6_layer(input, kernel_shape=[1, 1, in_channels, channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_branch2_conv1')
        branch2_conv2 = self.depthwise_conv_bn_layer(branch2_conv1, kernel_shape=[1, 1, channels, 1], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope_name + '_branch2_conv2')
        branch2_conv3 = self.conv_bn_relu6_layer(branch2_conv2, kernel_shape=[1, 1, channels, channels], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope_name + '_branch2_conv3')

        branch_concat = tf.concat([branch1_conv2, branch2_conv3], axis=3, name=scope_name + '_branch_concat')
        shuffle_features = self.channel_shuffle(branch_concat, groups=groups, scope_name=scope_name + '_shuffle')
        return shuffle_features
