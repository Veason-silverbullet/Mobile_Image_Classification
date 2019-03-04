import tensorflow as tf
import Model


class ResNet(Model.Model):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet, self).__init__(args, dtype, label_dim, initializer)
        self.dropout_rate = args.dropout_rate
        self.image_height = 224
        self.image_width = 224

    def conv_bn_layer(self, input, kernel_shape, strides, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding="SAME", name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        return bn

    def conv_bn_relu_layer(self, input, kernel_shape, strides, training, scope_name):
        conv_bn = self.conv_bn_layer(input, kernel_shape=kernel_shape, strides=strides, training=training, scope_name=scope_name)
        relu = tf.nn.relu(conv_bn, name=scope_name + '_conv_bn_relu')
        return relu

    def resnet_basic(self, input, out_channels, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        b_0 = self.conv_bn_relu_layer(input, kernel_shape=[3, 3, in_channels, out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b0')
        b = self.conv_bn_layer(b_0, kernel_shape=[3, 3, out_channels, out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b')

        p = b + input
        r = tf.nn.relu(p, name=scope_name + '_relu')
        return r

    def resnet_basic_inc(self, input, out_channels, strides, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        b_0 = self.conv_bn_relu_layer(input, kernel_shape=[3, 3, in_channels, out_channels], strides=strides, training=training, scope_name=scope_name + '_b0')
        b = self.conv_bn_layer(b_0, kernel_shape=[3, 3, out_channels, out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b')

        s = self.conv_bn_layer(input, kernel_shape=[1, 1, in_channels, out_channels], strides=strides, training=training, scope_name=scope_name + '_s')

        p = b + s
        r = tf.nn.relu(features=p, name=scope_name + '_relu')
        return r

    def resnet_bottleneck(self, input, out_channels, in_out_channels, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        b_0 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, in_out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b0')
        b_1 = self.conv_bn_relu_layer(b_0, kernel_shape=[3, 3, in_out_channels, in_out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b1')
        b = self.conv_bn_layer(b_1, kernel_shape=[1, 1, in_out_channels, out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b')

        p = b + input
        r = tf.nn.relu(features=p, name=scope_name + '_relu')
        return r

    def resnet_bottleneck_inc(self, input, out_channels, in_out_channels, stride_1x1, stride_3x3, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        b_0 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, in_out_channels], strides=stride_1x1, training=training, scope_name=scope_name + '_b0')
        b_1 = self.conv_bn_relu_layer(b_0, kernel_shape=[3, 3, in_out_channels, in_out_channels], strides=stride_3x3, training=training, scope_name=scope_name + '_b1')
        b = self.conv_bn_layer(b_1, kernel_shape=[1, 1, in_out_channels, out_channels], strides=[1, 1, 1, 1], training=training, scope_name=scope_name + '_b')

        strides = [1, stride_1x1[1] * stride_3x3[1], stride_1x1[2] * stride_3x3[2], 1]
        s = self.conv_bn_layer(input, kernel_shape=[1, 1, in_channels, out_channels], strides=strides, training=training, scope_name=scope_name + '_s')

        p = b + s
        r = tf.nn.relu(p, name=scope_name + '_relu')
        return r

    def resnet_basic_stack(self, input, n, out_channels, training, scope_name):
        resnet_basic_block = self.resnet_basic(input, out_channels=out_channels, training=training, scope_name=scope_name + '_resnet_basic_block0')
        for i in range(1, n):
            resnet_basic_block = self.resnet_basic(resnet_basic_block, out_channels=out_channels, training=training, scope_name=scope_name + '_resnet_basic_block' + str(i))
        return resnet_basic_block

    def resnet_bottleneck_stack(self, input, n, out_channels, in_out_channels, training, scope_name):
        resnet_bottleneck_block = self.resnet_bottleneck(input, out_channels=out_channels, in_out_channels=in_out_channels, training=training, scope_name=scope_name + '_resnet_bottleneck_block0')
        for i in range(1, n):
            resnet_bottleneck_block = self.resnet_bottleneck(resnet_bottleneck_block, out_channels=out_channels, in_out_channels=in_out_channels, training=training, scope_name=scope_name + '_resnet_bottleneck_block' + str(i))
        return resnet_bottleneck_block


class ResNet_18(ResNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet_18, self).__init__(args, dtype, label_dim, initializer)
        self.cmap = [64, 128, 256, 512]

    def model(self, images, training):
        conv1 = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, self.cmap[0]], strides=[1, 2, 2, 1], training=training, scope_name='conv1')
        max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')
        resnet_basic_stack1 = self.resnet_basic_stack(max_pool, n=2, out_channels=self.cmap[0], training=training, scope_name='resnet_basic_stack1')

        resnet_basic_inc1 = self.resnet_basic_inc(resnet_basic_stack1, out_channels=self.cmap[1], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc1')
        resnet_basic1 = self.resnet_basic(resnet_basic_inc1, out_channels=self.cmap[1], training=training, scope_name='resnet_basic1')

        resnet_basic_inc2 = self.resnet_basic_inc(resnet_basic1, out_channels=self.cmap[2], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc2')
        resnet_basic2 = self.resnet_basic(resnet_basic_inc2, out_channels=self.cmap[2], training=training, scope_name='resnet_basic2')

        resnet_basic_inc3 = self.resnet_basic_inc(resnet_basic2, out_channels=self.cmap[3], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc3')
        resnet_basic_stack2 = self.resnet_basic_stack(resnet_basic_inc3, n=2, out_channels=self.cmap[3], training=training, scope_name='resnet_basic_stack2')

        avg_pool = tf.nn.max_pool(resnet_basic_stack2, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.cmap[3], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ResNet_34(ResNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet_34, self).__init__(args, dtype, label_dim, initializer)
        self.cmap = [64, 128, 256, 512]
        self.num_layers = [3, 3, 5, 2]

    def model(self, images, training):
        conv1 = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, self.cmap[0]], strides=[1, 2, 2, 1], training=training, scope_name='conv1')
        max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')
        resnet_basic_stack1 = self.resnet_basic_stack(max_pool, n=self.num_layers[0], out_channels=self.cmap[0], training=training, scope_name='resnet_basic_stack1')

        resnet_basic_inc1 = self.resnet_basic_inc(resnet_basic_stack1, out_channels=self.cmap[1], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc1')
        resnet_basic_stack2 = self.resnet_basic_stack(resnet_basic_inc1, n=self.num_layers[1], out_channels=self.cmap[1], training=training, scope_name='resnet_basic_stack2')

        resnet_basic_inc2 = self.resnet_basic_inc(resnet_basic_stack2, out_channels=self.cmap[2], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc2')
        resnet_basic_stack3 = self.resnet_basic_stack(resnet_basic_inc2, n=self.num_layers[2], out_channels=self.cmap[2], training=training, scope_name='resnet_basic_stack3')

        resnet_basic_inc3 = self.resnet_basic_inc(resnet_basic_stack3, out_channels=self.cmap[3], strides=[1, 2, 2, 1], training=training, scope_name='resnet_basic_inc3')
        resnet_basic_stack4 = self.resnet_basic_stack(resnet_basic_inc3, n=self.num_layers[3], out_channels=self.cmap[3], training=training, scope_name='resnet_basic_stack4')

        avg_pool = tf.nn.max_pool(resnet_basic_stack4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.cmap[3], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ResNet_50(ResNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet_50, self).__init__(args, dtype, label_dim, initializer)
        self.cmap = [64, 128, 256, 512, 1024, 2048]
        self.num_layers = [3, 3, 5, 2]
        self.stride_1x1 = [1, 1, 1, 1]
        self.stride_3x3 = [1, 2, 2, 1]

    def model(self, images, training):
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, self.cmap[0]], strides=[1, 2, 2, 1], training=training, scope_name=scope.name)
            max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')

        resnet_bottleneck_inc1 = self.resnet_bottleneck_inc(max_pool, out_channels=self.cmap[2], in_out_channels=self.cmap[0], stride_1x1=[1, 1, 1, 1], stride_3x3=[1, 1, 1, 1], training=training, scope_name='resnet_basic_inc1')
        resnet_bottleneck_stack1 = self.resnet_bottleneck_stack(resnet_bottleneck_inc1, n=self.num_layers[0], out_channels=self.cmap[2], in_out_channels=self.cmap[0], training=training, scope_name='resnet_bottleneck_stack1')

        resnet_bottleneck_inc2 = self.resnet_bottleneck_inc(resnet_bottleneck_stack1, out_channels=self.cmap[3], in_out_channels=self.cmap[1], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc2')
        resnet_bottleneck_stack2 = self.resnet_bottleneck_stack(resnet_bottleneck_inc2, n=self.num_layers[1], out_channels=self.cmap[3], in_out_channels=self.cmap[1], training=training, scope_name='resnet_bottleneck_stack2')

        resnet_bottleneck_inc3 = self.resnet_bottleneck_inc(resnet_bottleneck_stack2, out_channels=self.cmap[4], in_out_channels=self.cmap[2], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc3')
        resnet_bottleneck_stack3 = self.resnet_bottleneck_stack(resnet_bottleneck_inc3, n=self.num_layers[2], out_channels=self.cmap[4], in_out_channels=self.cmap[2], training=training, scope_name='resnet_bottleneck_stack3')

        resnet_bottleneck_inc4 = self.resnet_bottleneck_inc(resnet_bottleneck_stack3, out_channels=self.cmap[5], in_out_channels=self.cmap[3], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc4')
        resnet_bottleneck_stack4 = self.resnet_bottleneck_stack(resnet_bottleneck_inc4, n=self.num_layers[3], out_channels=self.cmap[5], in_out_channels=self.cmap[3], training=training, scope_name='resnet_bottleneck_stack4')

        avg_pool = tf.nn.max_pool(resnet_bottleneck_stack4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.cmap[5], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ResNet_101(ResNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet_101, self).__init__(args, dtype, label_dim, initializer)
        self.cmap = [64, 128, 256, 512, 1024, 2048]
        self.num_layers = [2, 3, 22, 2]
        self.stride_1x1 = [1, 1, 1, 1]
        self.stride_3x3 = [1, 2, 2, 1]

    def model(self, images, training):
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, self.cmap[0]], strides=[1, 2, 2, 1], training=training, scope_name=scope.name)
            max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')

        resnet_bottleneck_inc1 = self.resnet_bottleneck_inc(max_pool, out_channels=self.cmap[2], in_out_channels=self.cmap[0], stride_1x1=[1, 1, 1, 1], stride_3x3=[1, 1, 1, 1], training=training, scope_name='resnet_basic_inc1')
        resnet_bottleneck_stack1 = self.resnet_bottleneck_stack(resnet_bottleneck_inc1, n=self.num_layers[0], out_channels=self.cmap[2], in_out_channels=self.cmap[0], training=training, scope_name='resnet_bottleneck_stack1')

        resnet_bottleneck_inc2 = self.resnet_bottleneck_inc(resnet_bottleneck_stack1, out_channels=self.cmap[3], in_out_channels=self.cmap[1], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc2')
        resnet_bottleneck_stack2 = self.resnet_bottleneck_stack(resnet_bottleneck_inc2, n=self.num_layers[1], out_channels=self.cmap[3], in_out_channels=self.cmap[1], training=training, scope_name='resnet_bottleneck_stack2')

        resnet_bottleneck_inc3 = self.resnet_bottleneck_inc(resnet_bottleneck_stack2, out_channels=self.cmap[4], in_out_channels=self.cmap[2], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc3')
        resnet_bottleneck_stack3 = self.resnet_bottleneck_stack(resnet_bottleneck_inc3, n=self.num_layers[2], out_channels=self.cmap[4], in_out_channels=self.cmap[2], training=training, scope_name='resnet_bottleneck_stack3')

        resnet_bottleneck_inc4 = self.resnet_bottleneck_inc(resnet_bottleneck_stack3, out_channels=self.cmap[5], in_out_channels=self.cmap[3], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name='resnet_basic_inc4')
        resnet_bottleneck_stack4 = self.resnet_bottleneck_stack(resnet_bottleneck_inc4, n=self.num_layers[3], out_channels=self.cmap[5], in_out_channels=self.cmap[3], training=training, scope_name='resnet_bottleneck_stack4')

        avg_pool = tf.nn.max_pool(resnet_bottleneck_stack4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.cmap[5], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z


class ResNet_152(ResNet):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(ResNet_152, self).__init__(args, dtype, label_dim, initializer)
        self.cmap = [64, 128, 256, 512, 1024, 2048]
        self.num_layers = [2, 7, 35, 2]
        self.stride_1x1 = [1, 1, 1, 1]
        self.stride_3x3 = [1, 2, 2, 1]

    def model(self, images, training):
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_bn_relu_layer(images, kernel_shape=[7, 7, 3, self.cmap[0]], strides=[1, 2, 2, 1], training=training, scope_name=scope.name)
            max_pool = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name='max_pool')

        with tf.variable_scope('resnet_block1') as scope:
            resnet_bottleneck_inc1 = self.resnet_bottleneck_inc(max_pool, out_channels=self.cmap[2], in_out_channels=self.cmap[0], stride_1x1=[1, 1, 1, 1], stride_3x3=[1, 1, 1, 1], training=training, scope_name=scope.name + '_basic_inc1')
            resnet_bottleneck_stack1 = self.resnet_bottleneck_stack(resnet_bottleneck_inc1, n=self.num_layers[0], out_channels=self.cmap[2], in_out_channels=self.cmap[0], training=training, scope_name=scope.name + '_bottleneck_stack1')

        with tf.variable_scope('resnet_block2') as scope:
            resnet_bottleneck_inc2 = self.resnet_bottleneck_inc(resnet_bottleneck_stack1, out_channels=self.cmap[3], in_out_channels=self.cmap[1], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name=scope.name + '_basic_inc2')
            resnet_bottleneck_stack2 = self.resnet_bottleneck_stack(resnet_bottleneck_inc2, n=self.num_layers[1], out_channels=self.cmap[3], in_out_channels=self.cmap[1], training=training, scope_name=scope.name + '_bottleneck_stack2')

        with tf.variable_scope('resnet_block3') as scope:
            resnet_bottleneck_inc3 = self.resnet_bottleneck_inc(resnet_bottleneck_stack2, out_channels=self.cmap[4], in_out_channels=self.cmap[2], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name=scope.name + '_basic_inc3')
            resnet_bottleneck_stack3 = self.resnet_bottleneck_stack(resnet_bottleneck_inc3, n=self.num_layers[2], out_channels=self.cmap[4], in_out_channels=self.cmap[2], training=training, scope_name=scope.name + '_bottleneck_stack3')

        with tf.variable_scope('resnet_block4') as scope:
            resnet_bottleneck_inc4 = self.resnet_bottleneck_inc(resnet_bottleneck_stack3, out_channels=self.cmap[5], in_out_channels=self.cmap[3], stride_1x1=self.stride_1x1, stride_3x3=self.stride_3x3, training=training, scope_name=scope.name + '_basic_inc4')
            resnet_bottleneck_stack4 = self.resnet_bottleneck_stack(resnet_bottleneck_inc4, n=self.num_layers[3], out_channels=self.cmap[5], in_out_channels=self.cmap[3], training=training, scope_name=scope.name + '_bottleneck_stack4')

        avg_pool = tf.nn.max_pool(resnet_bottleneck_stack4, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID", name='avg_pool')
        w = tf.get_variable(shape=[self.cmap[5], self.label_dim], initializer=tf.random_normal_initializer(stddev=0.01), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z = tf.add(tf.matmul(tf.reshape(avg_pool, shape=[avg_pool.get_shape()[0], -1]), w), b, name='z')

        return z
