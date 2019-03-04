import tensorflow as tf
import Model


class Inception_V3(Model.Model):

    def __init__(self, args, dtype, label_dim, initializer=None):
        super(Inception_V3, self).__init__(args, dtype, label_dim, initializer)
        self.dropout_rate = args.dropout_rate
        self.image_height = 299
        self.image_width = 299

    def conv_bn_relu_layer(self, input, kernel_shape, strides, padding, training, scope_name):
        kernel = tf.get_variable(shape=kernel_shape, initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name=scope_name + '_kernel')
        conv = tf.nn.conv2d(input, filter=kernel, strides=strides, padding=padding, name=scope_name + '_conv')
        bn = tf.layers.batch_normalization(conv, training=training, name=scope_name + '_conv_bn')
        relu = tf.nn.relu(bn, name=scope_name + '_conv_bn_relu')
        return relu

    def inception_block1(self, input, out_channels_1x1, out_channels_5x5, out_channels_3x3_3x3, out_channels_pool, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        branch_1x1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_1x1], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_1x1')

        branch_5x5_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_5x5[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_5x5_1')
        branch_5x5 = self.conv_bn_relu_layer(branch_5x5_1, kernel_shape=[5, 5, out_channels_5x5[0], out_channels_5x5[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_5x5')

        branch_3x3_3x3_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_3x3_3x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_1')
        branch_3x3_3x3_2 = self.conv_bn_relu_layer(branch_3x3_3x3_1, kernel_shape=[3, 3, out_channels_3x3_3x3[0], out_channels_3x3_3x3[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_2')
        branch_3x3_3x3 = self.conv_bn_relu_layer(branch_3x3_3x3_2, kernel_shape=[3, 3, out_channels_3x3_3x3[1], out_channels_3x3_3x3[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3')

        branch_pool_1 = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name=scope_name + 'branch_pool_1')
        branch_pool = self.conv_bn_relu_layer(branch_pool_1, kernel_shape=[1, 1, in_channels, out_channels_pool], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_pool')

        out = tf.concat([branch_1x1, branch_5x5, branch_3x3_3x3, branch_pool], axis=3, name=scope_name + '_concat')
        return out

    def inception_block2(self, input, out_channels_3x3, out_channels_3x3_3x3, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        branch_3x3 = self.conv_bn_relu_layer(input, kernel_shape=[3, 3, in_channels, out_channels_3x3], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope_name + '_branch_3x3')

        branch_3x3_3x3_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_3x3_3x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_1')
        branch_3x3_3x3_2 = self.conv_bn_relu_layer(branch_3x3_3x3_1, kernel_shape=[3, 3, out_channels_3x3_3x3[0], out_channels_3x3_3x3[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_2')
        branch_3x3_3x3 = self.conv_bn_relu_layer(branch_3x3_3x3_2, kernel_shape=[3, 3, out_channels_3x3_3x3[1], out_channels_3x3_3x3[2]], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope_name + '_branch_3x3_3x3')

        branch_pool = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name=scope_name + 'branch_pool')

        out = tf.concat(values=[branch_3x3, branch_3x3_3x3, branch_pool], axis=3, name=scope_name + '_concat')
        return out

    def inception_block3(self, input, out_channels_1x1, out_channels_7x7, out_channels_7x7_7x7, out_channels_pool, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        branch_1x1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_1x1], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_1x1')

        branch_7x7_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_7x7[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_1')
        branch_7x7_2 = self.conv_bn_relu_layer(branch_7x7_1, kernel_shape=[1, 7, out_channels_7x7[0], out_channels_7x7[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_2')
        branch_7x7 = self.conv_bn_relu_layer(branch_7x7_2, kernel_shape=[7, 1, out_channels_7x7[1], out_channels_7x7[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7')

        branch_7x7_7x7_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_7x7_7x7[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_7x7_1')
        branch_7x7_7x7_2 = self.conv_bn_relu_layer(branch_7x7_7x7_1, kernel_shape=[7, 1, out_channels_7x7_7x7[0], out_channels_7x7_7x7[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_7x7_2')
        branch_7x7_7x7_3 = self.conv_bn_relu_layer(branch_7x7_7x7_2, kernel_shape=[1, 7, out_channels_7x7_7x7[1], out_channels_7x7_7x7[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_7x7_3')
        branch_7x7_7x7_4 = self.conv_bn_relu_layer(branch_7x7_7x7_3, kernel_shape=[7, 1, out_channels_7x7_7x7[2], out_channels_7x7_7x7[3]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_7x7_4')
        branch_7x7_7x7 = self.conv_bn_relu_layer(branch_7x7_7x7_4, kernel_shape=[1, 7, out_channels_7x7_7x7[3], out_channels_7x7_7x7[4]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7_7x7')

        branch_pool_1 = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name=scope_name + 'branch_pool_1')
        branch_pool = self.conv_bn_relu_layer(branch_pool_1, kernel_shape=[1, 1, in_channels, out_channels_pool], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_pool')

        out = tf.concat(values=[branch_1x1, branch_7x7, branch_7x7_7x7, branch_pool], axis=3, name=scope_name + '_concat')
        return out

    def inception_block4(self, input, out_channels_3x3, out_channels_7x7x3, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        branch_3x3_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_3x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_1')
        branch_3x3 = self.conv_bn_relu_layer(branch_3x3_1, kernel_shape=[3, 3, out_channels_3x3[0], out_channels_3x3[1]], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope_name + '_branch_3x3')

        branch_7x7x3_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_7x7x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7x3_1')
        branch_7x7x3_2 = self.conv_bn_relu_layer(branch_7x7x3_1, kernel_shape=[1, 7, out_channels_7x7x3[0], out_channels_7x7x3[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7x3_2')
        branch_7x7x3_3 = self.conv_bn_relu_layer(branch_7x7x3_2, kernel_shape=[7, 1, out_channels_7x7x3[1], out_channels_7x7x3[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_7x7x3_3')
        branch_7x7x3 = self.conv_bn_relu_layer(branch_7x7x3_3, kernel_shape=[3, 3, out_channels_7x7x3[2], out_channels_7x7x3[3]], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope_name + '_branch_7x7x3')

        branch_pool = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name=scope_name + 'branch_pool')

        out = tf.concat(values=[branch_3x3, branch_7x7x3, branch_pool], axis=3, name=scope_name + '_concat')
        return out

    def inception_block5(self, input, out_channels_1x1, out_channels_3x3, out_channels_3x3_3x3, out_channels_pool, training, scope_name):
        in_channels = input.shape.as_list()[-1]
        branch_1x1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_1x1], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_1x1')

        branch_3x3_0 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_3x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_0')
        branch_3x3_1 = self.conv_bn_relu_layer(branch_3x3_0, kernel_shape=[1, 3, out_channels_3x3[0], out_channels_3x3[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_1')
        branch_3x3_2 = self.conv_bn_relu_layer(branch_3x3_1, kernel_shape=[3, 1, out_channels_3x3[1], out_channels_3x3[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_2')
        branch_3x3 = tf.concat([branch_3x3_1, branch_3x3_2], axis=3, name=scope_name + '_branch_3x3')

        branch_3x3_3x3_0_1 = self.conv_bn_relu_layer(input, kernel_shape=[1, 1, in_channels, out_channels_3x3_3x3[0]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_0_1')
        branch_3x3_3x3_0 = self.conv_bn_relu_layer(branch_3x3_3x3_0_1, kernel_shape=[3, 3, out_channels_3x3_3x3[0], out_channels_3x3_3x3[1]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_0')

        branch_3x3_3x3_1 = self.conv_bn_relu_layer(branch_3x3_3x3_0, kernel_shape=[1, 3, out_channels_3x3_3x3[1], out_channels_3x3_3x3[2]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_1')
        branch_3x3_3x3_2 = self.conv_bn_relu_layer(branch_3x3_3x3_1, kernel_shape=[3, 1, out_channels_3x3_3x3[1], out_channels_3x3_3x3[3]], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_3x3_3x3_2')
        branch_3x3_3x3 = tf.concat([branch_3x3_3x3_1, branch_3x3_3x3_2], axis=3, name=scope_name + '_branch_3x3_3x3')

        branch_pool_1 = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME", name=scope_name + 'branch_pool_1')
        branch_pool = self.conv_bn_relu_layer(branch_pool_1, kernel_shape=[1, 1, in_channels, out_channels_pool], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope_name + '_branch_pool')

        out = tf.concat([branch_1x1, branch_3x3, branch_3x3_3x3, branch_pool], axis=3, name=scope_name + '_concat')
        return out

    def model(self, images, training):
        with tf.variable_scope('conv1') as scope:
            conv1 = self.conv_bn_relu_layer(images, kernel_shape=[3, 3, 3, 32], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope.name)

        with tf.variable_scope('conv2') as scope:
            conv2 = self.conv_bn_relu_layer(conv1, kernel_shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope.name)

        with tf.variable_scope('conv3') as scope:
            conv3 = self.conv_bn_relu_layer(conv2, kernel_shape=[3, 3, 32, 64], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name=scope.name)

        with tf.variable_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)

        with tf.variable_scope('conv4') as scope:
            conv4 = self.conv_bn_relu_layer(pool1, kernel_shape=[3, 3, 64, 80], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name=scope.name)

        with tf.variable_scope('conv5') as scope:
            conv5 = self.conv_bn_relu_layer(conv4, kernel_shape=[3, 3, 80, 192], strides=[1, 2, 2, 1], padding="VALID", training=training, scope_name=scope.name)

        mixed_1 = self.inception_block1(conv5, out_channels_1x1=64, out_channels_5x5=[48, 64], out_channels_3x3_3x3=[64, 96, 96], out_channels_pool=32, training=training, scope_name='mixed_1')
        mixed_2 = self.inception_block1(mixed_1, out_channels_1x1=64, out_channels_5x5=[48, 64], out_channels_3x3_3x3=[64, 96, 96], out_channels_pool=64, training=training, scope_name='mixed_2')
        mixed_3 = self.inception_block1(mixed_2, out_channels_1x1=64, out_channels_5x5=[48, 64], out_channels_3x3_3x3=[64, 96, 96], out_channels_pool=64, training=training, scope_name='mixed_3')
        mixed_4 = self.inception_block2(mixed_3, out_channels_3x3=384, out_channels_3x3_3x3=[64, 96, 96], training=training, scope_name='mixed_4')
        mixed_5 = self.inception_block3(mixed_4, out_channels_1x1=192, out_channels_7x7=[128, 128, 192], out_channels_7x7_7x7=[128, 128, 128, 128, 192], out_channels_pool=192, training=training, scope_name='mixed_5')
        mixed_6 = self.inception_block3(mixed_5, out_channels_1x1=192, out_channels_7x7=[160, 160, 192], out_channels_7x7_7x7=[160, 160, 160, 160, 192], out_channels_pool=192, training=training, scope_name='mixed_6')
        mixed_7 = self.inception_block3(mixed_6, out_channels_1x1=192, out_channels_7x7=[160, 160, 192], out_channels_7x7_7x7=[160, 160, 160, 160, 192], out_channels_pool=192, training=training, scope_name='mixed_7')
        mixed_8 = self.inception_block3(mixed_7, out_channels_1x1=192, out_channels_7x7=[192, 192, 192], out_channels_7x7_7x7=[192, 192, 192, 192, 192], out_channels_pool=192, training=training, scope_name='mixed_8')
        mixed_9 = self.inception_block4(mixed_8, out_channels_3x3=[192, 320], out_channels_7x7x3=[192, 192, 192, 192], training=training, scope_name='mixed_9')
        mixed_10 = self.inception_block5(mixed_9, out_channels_1x1=320, out_channels_3x3=[384, 384, 384], out_channels_3x3_3x3=[448, 384, 384, 384], out_channels_pool=192, training=training, scope_name='mixed_10')
        mixed_11 = self.inception_block5(mixed_10, out_channels_1x1=320, out_channels_3x3=[384, 384, 384], out_channels_3x3_3x3=[448, 384, 384, 384], out_channels_pool=192, training=training, scope_name='mixed_11')

        pool3 = tf.nn.avg_pool(mixed_11, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID", name='pool3')
        drop = tf.nn.dropout(pool3, keep_prob=1 - self.dropout_rate, name='dropout')
        w = tf.get_variable(shape=[2048, self.label_dim], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name='weight')
        b = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias')
        z0 = tf.add(tf.matmul(tf.reshape(drop, shape=[drop.get_shape()[0], -1]), w), b, name='z0')

        aux_pool = tf.nn.avg_pool(mixed_8, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1], padding="VALID", name='aux_pool')
        aux_conv_1 = self.conv_bn_relu_layer(aux_pool, kernel_shape=[1, 1, 768, 128], strides=[1, 1, 1, 1], padding="SAME", training=training, scope_name='aux_conv_1')
        aux_conv_2 = self.conv_bn_relu_layer(aux_conv_1, kernel_shape=[5, 5, 128, 768], strides=[1, 1, 1, 1], padding="VALID", training=training, scope_name='aux_conv_2')
        w_aux = tf.get_variable(shape=[768, self.label_dim], initializer=tf.contrib.layers.variance_scaling_initializer(), dtype=self.dtype, name='weight_aux')
        b_aux = tf.get_variable(shape=[self.label_dim], initializer=tf.zeros_initializer(), dtype=self.dtype, name='bias_aux')
        aux = tf.add(tf.matmul(tf.reshape(aux_conv_2, shape=[aux_conv_2.get_shape()[0], -1]), w_aux), b_aux, name='aux')

        z = tf.add(z0, aux, name='z')

        return z
