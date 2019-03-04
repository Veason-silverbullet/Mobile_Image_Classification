import tensorflow as tf
from _datetime import datetime
import time
import math
import os


class Model:

    def __init__(self, args, dtype, label_dim, initializer):
        self.args = args
        self.dtype = dtype
        self.label_dim = label_dim
        self.initializer = initializer if initializer is not None else tf.initializers.zeros()
        self.batch_size = self.args.batch_size
        self.steps = int(math.floor(self.args.train_image_num * self.args.epoch / self.args.batch_size))
        self.optimizer = self.args.optimizer
        self.lr_policy = self.args.lr_policy
        self.init_learning_rate = self.args.init_learning_rate
        self.end_learning_rate = self.args.end_learning_rate
        self.lr_decay_power = self.args.lr_decay_power
        self.momentum = self.args.momentum
        self.useNAG = self.args.useNAG
        self.clipping_gradient = self.args.clipping_gradient
        self.learning_rate_decay_epoch = 2
        self.learning_rate_decay_steps = int(math.floor(self.args.train_image_num * self.learning_rate_decay_epoch / self.args.batch_size))
        self.learning_rate_decay = 0.98
        self.train_image_num = self.args.train_image_num
        self.L2Reg_weight = self.args.L2Reg_weight
        self.thread_num = 4

    def get_cifar_mini_batch_data(self, data_path, batch_size, shuffle=True):
        file_name_queue = tf.train.string_input_producer([data_path], shuffle=True, num_epochs=self.args.epoch, name='file_name_queue')
        label_bytes = 2
        image_bytes = 3072
        record_bytes = label_bytes + image_bytes
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        key, value = reader.read(file_name_queue)
        data = tf.decode_raw(value, tf.uint8)
        label = tf.cast(tf.strided_slice(data, [1], [2]), tf.int64)
        image = tf.strided_slice(data, [2], [record_bytes])
        image = tf.reshape(image, shape=[3, 32, 32])
        image = tf.transpose(image, perm=[1, 2, 0])
        resized_image = tf.image.resize_image_with_crop_or_pad(image, self.image_height, self.image_width)
        float_image = tf.cast(resized_image, self.dtype)
        normalized_image = (float_image - 128.0) / 128.0
        label.set_shape([1])
        normalized_image.set_shape([self.image_height, self.image_width, 3])
        min_queue_examples = int(self.train_image_num * 0.1)
        if shuffle:
            images_batch = tf.train.shuffle_batch(
                [normalized_image, label],
                batch_size=batch_size,
                num_threads=self.thread_num,
                capacity=min_queue_examples + 16 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images_batch = tf.train.batch(
                [normalized_image, label],
                batch_size=batch_size,
                num_threads=self.thread_num,
                capacity=min_queue_examples + 3 * batch_size)
        return images_batch

    def get_imagenet_mini_batch_data(self, label_file, batch_size, shuffle=True):
        if label_file is None or os.path.exists(label_file) is False:
            raise Exception('Label file not found.')
        images_path = []
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if line != '':
                    image_batch, label = line.split(' ')
                    images_path.append(self.args.train_dir + '/' + image_batch)
                    labels.append(label)

        file_name_queue = tf.train.string_input_producer([images_path], shuffle=True, num_epochs=self.args.epoch, name='file_name_queue')
        reader = tf.whole_file_reader()
        _, image = reader.read(file_name_queue)
        image = tf.image.decode_jpeg(image)
        data = tf.decode_raw(image, tf.uint8)
        label = tf.cast(tf.strided_slice(data, [1], [2]), tf.int64)
        resized_image = tf.image.resize_image_with_crop_or_pad(image, self.image_height, self.image_width)
        float_image = tf.cast(resized_image, self.dtype)
        normalized_image = (float_image - 128.0) / 128.0
        label.set_shape([1])
        normalized_image.set_shape([self.image_height, self.image_width, 3])
        min_queue_examples = int(self.train_image_num * 0.1)
        if shuffle:
            images_batch = tf.train.shuffle_batch(
                [normalized_image, label],
                batch_size=batch_size,
                num_threads=self.thread_num,
                capacity=min_queue_examples + 16 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images_batch = tf.train.batch(
                [normalized_image, label],
                batch_size=batch_size,
                num_threads=self.thread_num,
                capacity=min_queue_examples + 3 * batch_size)
        return images_batch

    def loss(self, labels, logits, L2Reg_weight=None):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_loss')
        loss_mean = tf.reduce_mean(loss, name='cross_entropy_loss_mean')
        if L2Reg_weight is not None:
            reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(L2Reg_weight), tf.trainable_variables())
            loss_mean = loss_mean + reg
        return loss_mean

    def train(self):
        step_to_show_result = self.args.step_to_show_result
        batch_size = self.args.batch_size
        gs = 0

        with tf.Graph().as_default() as graph:
            global_step = tf.train.get_or_create_global_step()
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            with tf.device('/cpu:0'):
                if self.args.dataset == 'Cifar-100' or self.args.dataset == 'cifar-100':
                    images_batch, labels_batch = self.get_cifar_mini_batch_data(data_path=self.args.train_dir + '/train.bin', batch_size=batch_size, shuffle=True)
                elif self.args.dataset == 'ImageNet' or self.args.dataset == 'imageNet' or self.args.dataset == 'imagenet':
                    images_batch, labels_batch = self.get_imagenet_mini_batch_data(label_file=self.args.train_dir + '/train.bin', batch_size=batch_size, shuffle=True)
                else:
                    raise Exception('Unexpected dataset \"%s\". Dataset must be [ImageNet | Cifar-100]', self.args.dataset)
            labels_batch = tf.reshape(labels_batch, [batch_size])
            training = tf.placeholder(dtype=tf.bool, name='training')

            # Forward
            with tf.variable_scope('loss'):
                logits = self.model(images_batch, training)
                loss = self.loss(labels=labels_batch, logits=logits, L2Reg_weight=self.L2Reg_weight)
                summaries.append(tf.summary.scalar('loss', loss))
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), labels_batch)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                summaries.append(tf.summary.scalar('accuracy', accuracy))

            # Backward
            with tf.variable_scope('gradient'):
                if self.lr_policy == 'exponential_decay':
                    learning_rate = tf.train.exponential_decay(learning_rate=self.init_learning_rate, global_step=global_step, decay_steps=self.learning_rate_decay_steps, decay_rate=self.learning_rate_decay, staircase=True, name='learning_rate')
                elif self.lr_policy == 'polynomial_decay':
                    learning_rate = tf.train.polynomial_decay(learning_rate=self.init_learning_rate, global_step=global_step, decay_steps=self.learning_rate_decay_steps, end_learning_rate=self.end_learning_rate, power=self.lr_decay_power, name='learning_rate')
                else:
                    raise Exception('Unexpected lr_policy \"%s\". Learning rate policy must be [exponential_decay | polynomial_decay]', self.lr_policy)
                summaries.append(tf.summary.scalar('learning_rate', learning_rate))
                if self.optimizer == 'sgd' or self.optimizer == 'SGD':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                elif self.optimizer == 'adam' or self.optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                elif self.optimizer == 'momentum' or self.optimizer == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum, use_nesterov=self.useNAG)
                elif self.optimizer == 'rmsprop' or self.optimizer == 'RMSProp':
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=self.momentum, decay=0.9)
                else:
                    raise Exception('Unexpected optimizer \"%s\". Optimizer must be [sgd | adam | momentum | rmsprop]', self.optimizer)
                gradients = optimizer.compute_gradients(loss)
                if self.args.gradient_clipping:
                    gradients = [(tf.clip_by_norm(grad, 1), var) for grad, var in gradients]
                train_op = optimizer.apply_gradients(gradients, global_step=global_step)

            summary_op = tf.summary.merge(summaries)
            writer = tf.summary.FileWriter(self.args.summary_dir)
            writer.add_graph(graph=graph)

            class LoggerHook(tf.train.SessionRunHook):
                def __init__(self):
                    self._step = None
                    self._start_time = None

                def begin(self):
                    self._step = -1
                    self._start_time = time.time()

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs([loss, accuracy])  # Asks for loss value.

                def after_run(self, run_context, run_values):
                    if self._step % step_to_show_result == 0:
                        current_time = time.time()
                        duration = current_time - self._start_time
                        self._start_time = current_time
                        loss_value, accuracy_value = run_values.results
                        examples_per_sec = step_to_show_result * batch_size / duration
                        sec_per_batch = float(duration / step_to_show_result)
                        print('%s: step %d, loss = %.2f, acc = %.2lf (%.1f examples/sec; %.3f sec/batch)' %
                              (datetime.now(), self._step, loss_value, accuracy_value, examples_per_sec, sec_per_batch))

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.args.checkpoint_dir,
                    hooks=[tf.train.StopAtStepHook(last_step=self.steps), tf.train.NanTensorHook(loss), LoggerHook()],
                    config=tf.ConfigProto(log_device_placement=self.args.log_device_placement),
                    save_checkpoint_secs=1800
            ) as mon_sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)
                while not mon_sess.should_stop():
                    _, summary = mon_sess.run([train_op, summary_op], feed_dict={training: True})
                    gs += 1
                    if gs % step_to_show_result == 0:
                        writer.add_summary(summary=summary, global_step=gs)
                coord.request_stop()
                coord.join(threads)
