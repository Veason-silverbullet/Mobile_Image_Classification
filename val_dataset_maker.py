import tensorflow as tf
import numpy as np
from PIL import Image


def make_example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))


def gen_tfrecord(trainrecords):
    file_num = 0
    writer = tf.python_io.TFRecordWriter("val.tfrecord")
    for record in trainrecords:
        file_num += 1
        fields = record.strip('\n').split(',')
        image = Image.open(fields[0])
        if image.mode == 'RGB':
            image = image.resize((256, 256))
            image_bytes = image.tobytes()
            label = np.array(int(fields[1]))
            ex = make_example(image_bytes, label)
            writer.write(ex.SerializeToString())
        else:
            raise Exception('')
    writer.close()


if __name__ == '__main__':
    with open('validation_label.csv', 'r') as trainfile:
        trainrecords = trainfile.readlines()
    gen_tfrecord(trainrecords)
