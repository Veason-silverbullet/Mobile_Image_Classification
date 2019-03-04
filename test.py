import tensorflow as tf

batch_size = 8
file_name_list = []
with open('file_name_list.txt', 'r') as f:
    for file_name in f:
        if file_name != '':
            file_name_list.append(('E:/imageNet/' + file_name).strip('\n'))
file_name_queue = tf.train.string_input_producer(file_name_list)
reader = tf.WholeFileReader()
key, raw_images = reader.read(file_name_queue)
images = tf.image.decode_jpeg(raw_images, channels=3)
resized_images = tf.image.resize_image_with_crop_or_pad(images, 299, 299)
resized_float_images = tf.cast(resized_images, tf.float32)
resized_float_norm_images = resized_float_images / 128.0 - 128.0
resized_images.set_shape([299, 299, 3])
min_queue_examples = 1000
images_batch = tf.train.shuffle_batch(
                [resized_float_norm_images],
                batch_size=batch_size,
                num_threads=8,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(images_batch)
    coord.request_stop()
    coord.join(threads)
