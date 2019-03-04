import tensorflow as tf
import Model


class ResNeXt(Model.Model):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNeXt, self).__init__(args, dtype, label_dim, initializer)
        self.cardinality = 32
        self.image_height = 224
        self.image_width = 224
        self.use_conv = True

    def conv_bn_layer(self, input, kernel_shape, strides, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding="SAME", name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def conv_bn_relu_layer(self, input, kernel_shape, strides, training, scope_name):
        conv_bn = self.conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, training=training, scope_name=scope_name)
        relu = tf.nn.relu(conv_bn, name=scope_name + '_conv_bn_relu')
        return relu

    def depthwise_conv_bn_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        depthwise_conv = tf.nn.depthwise_conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_depthwise_conv')
        depthwise_conv_bn = tf.layers.batch_normalization(depthwise_conv, training=training, name=scope_name + '_depthwise_conv_bn')
        return depthwise_conv_bn

    def depthwise_conv_bn_relu_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        depthwise_conv_bn = self.depthwise_conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, padding=padding, training=training, scope_name=scope_name)
        depthwise_conv_bn_relu = tf.nn.relu(depthwise_conv_bn, name=scope_name + '_depthwise_conv_bn_relu')
        return depthwise_conv_bn_relu

    def group_conv_bn_relu(self, input, kernel_shape, strides, cardinality, out_channels, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        assert in_channels % cardinality == 0 and out_channels % cardinality == 0 and len(kernel_shape) == 2
        group_channels = in_channels // cardinality
        group_filter_num = out_channels // cardinality

        kernel = tf.get_variable(shape=kernel_shape + [in_channels, group_filter_num], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + 'depthwise_kernel')
        group_conv = tf.nn.depthwise_conv2d(input, filter=kernel, strides=strides, padding="SAME", name=scope_name + '_group_conv')
        group_conv_shape = group_conv.shape.as_list()
        group_conv = tf.reshape(group_conv, group_conv_shape[0:3] + [cardinality, group_channels, group_filter_num])
        group_conv = tf.reduce_sum(group_conv, axis=4)
        group_conv = tf.reshape(group_conv, group_conv_shape[0:3] + [out_channels])

        group_conv_bn = tf.layers.batch_normalization(group_conv, training=training, name=scope_name + '_group_conv_bn')
        group_conv_bn_relu = tf.nn.relu(group_conv_bn, name=scope_name + '_group_conv_bn_relu')

        return group_conv_bn_relu

    def block(self, input, expand_channel, strides, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        if expand_channel:
            conv1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, in_channels // 2], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_conv1')
            conv2 = self.group_conv_bn_relu(conv1, kernel_shape=[3, 3], strides=strides, cardinality=self.cardinality, out_channels=in_channels // 2, training=training, scope_name=scope_name + '_conv2')
            conv3 = self.conv_bn_relu_layer(conv2, kernel_shape=[1, 1, in_channels // 2, in_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_conv3')
            if self.use_conv or strides[1] != 1 or strides[2] != 1:
                res = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, in_channels], strides=strides, training=training, scope_name=scope_name + '_res')
                out = tf.nn.relu(tf.concat([res, conv3], axis=3), name=scope_name + '_out')
            else:
                out = tf.nn.relu(tf.concat([input, conv3], axis=3), name=scope_name + '_out')
        else:
            branch1, branch2 = tf.split(input, num_or_size_splits=2, axis=3, name=scope_name + '_branch_split')
            branch1_in_channels = branch1.shape.as_list()[-1]
            branch2_in_channels = branch2.shape.as_list()[-1]
            assert branch1_in_channels == branch2_in_channels
            conv1 = self.conv_bn_relu_layer(branch2, kernel_shape=[1, 1, branch2_in_channels, in_channels // 4], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_conv1')
            conv2 = self.group_conv_bn_relu(conv1, kernel_shape=[3, 3], strides=strides, cardinality=self.cardinality, out_channels=in_channels // 4, training=training, scope_name=scope_name + '_conv2')
            conv3 = self.conv_bn_relu_layer(conv2, kernel_shape=[1, 1, in_channels // 4, branch2_in_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_conv3')
            if self.use_conv or strides[1] != 1 or strides[2] != 1:
                res = self.conv_bn_relu_layer(branch1, kernel_shape=[1, 1, branch1_in_channels, branch1_in_channels], strides=strides, training=training, scope_name=scope_name + '_res')
                out = tf.nn.relu(tf.concat([res, conv3], axis=3), name=scope_name + '_out')
            else:
                out = tf.nn.relu(tf.concat([branch1, conv3], axis=3), name=scope_name + '_out')
        return out

    def block_stack(self, input, n, strides, training, scope_name):
        net = self.block(input, expand_channel=True, strides=strides, training=training, scope_name=scope_name + '_0')
        for i in range(1, n):
            net = self.block(net, expand_channel=False, strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_' + str(i))
        return net

    def model(self, images, training):
        conv = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, 64], strides=[1, 2, 2, 1], training=training, scope_name='conv')
        max_pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')

        block1 = self.block_stack(max_pool, n=self.block_depth[0], strides=[1, 1, 1, 1], training=training, scope_name='block1')
        block2 = self.block_stack(block1, n=self.block_depth[1], strides=[1, 2, 2, 1], training=training, scope_name='block2')
        block3 = self.block_stack(block2, n=self.block_depth[2], strides=[1, 2, 2, 1], training=training, scope_name='block3')
        block4 = self.block_stack(block3, n=self.block_depth[3], strides=[1, 2, 2, 1], training=training, scope_name='block4')

        avg_pool = tf.nn.max_pool(block4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[1024, self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ResNeXt_50(ResNeXt):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNeXt_50, self).__init__(args, dtype, label_dim, initializer)
        self.block_depth = [3, 4, 6, 3]


class ResNeXt_101(ResNeXt):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNeXt_101, self).__init__(args, dtype, label_dim, initializer)
        self.block_depth = [3, 4, 23, 3]


class ResNeXt_152(ResNeXt):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNeXt_152, self).__init__(args, dtype, label_dim, initializer)
        self.block_depth = [3, 8, 36, 3]
