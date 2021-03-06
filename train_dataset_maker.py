import tensorflow as tf
from multiprocessing import Process, Queue
from _datetime import time
import numpy as np
from PIL import Image


def make_example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }))


def gen_tfrecord(trainrecords, queue):
    pre = ''
    file_num = 0
    writer = None

    for record in trainrecords:
        if record.strip('\n') == '':
            continue
        s = record.split('/')
        if pre != s[1]:
            if writer is not None:
                writer.close()
            writer = tf.python_io.TFRecordWriter("train/" + s[1] + ".tfrecord")
            pre = s[1]

        fields = record.strip('\n').split(',')
        image = Image.open(fields[0])
        if image.mode == 'RGB':
            image = image.resize((256, 256))
            image_bytes = image.tobytes()
            label = np.array(int(fields[1]))
            ex = make_example(image_bytes, label)
            writer.write(ex.SerializeToString())
            file_num += 1
            if file_num % 100 == 0:
                queue.put(file_num)
    writer.close()


if __name__ == '__main__':
    trainrecords = [[] for _ in range(10)]
    with open('train_label.csv', 'r') as in_file:
        pre = ''
        cnt = 0
        for line in in_file:
            s = line.split('/')
            if s[1] != pre:
                pre = s[1]
                cnt += 1
            trainrecords[(cnt - 1) // 100].append(line)

    q0 = Queue()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q5 = Queue()
    q6 = Queue()
    q7 = Queue()
    q8 = Queue()
    q9 = Queue()
    p0 = Process(target=gen_tfrecord, args=(trainrecords[0], q0,))
    p1 = Process(target=gen_tfrecord, args=(trainrecords[1], q1,))
    p2 = Process(target=gen_tfrecord, args=(trainrecords[2], q2,))
    p3 = Process(target=gen_tfrecord, args=(trainrecords[3], q3,))
    p4 = Process(target=gen_tfrecord, args=(trainrecords[4], q4,))
    p5 = Process(target=gen_tfrecord, args=(trainrecords[5], q5,))
    p6 = Process(target=gen_tfrecord, args=(trainrecords[6], q6,))
    p7 = Process(target=gen_tfrecord, args=(trainrecords[7], q7,))
    p8 = Process(target=gen_tfrecord, args=(trainrecords[8], q8,))
    p9 = Process(target=gen_tfrecord, args=(trainrecords[9], q9,))

    msg0 = 0
    msg1 = 0
    msg2 = 0
    msg3 = 0
    msg4 = 0
    msg5 = 0
    msg6 = 0
    msg7 = 0
    msg8 = 0
    msg9 = 0

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()

    while (True):
        try:
            if p0.is_alive():
                msg0 = q0.get()
            if p1.is_alive():
                msg1 = q1.get()
            if p2.is_alive():
                msg2 = q2.get()
            if p3.is_alive():
                msg3 = q3.get()
            if p4.is_alive():
                msg4 = q4.get()
            if p5.is_alive():
                msg5 = q5.get()
            if p6.is_alive():
                msg6 = q6.get()
            if p7.is_alive():
                msg7 = q7.get()
            if p8.is_alive():
                msg8 = q8.get()
            if p9.is_alive():
                msg9 = q9.get()
            print('P0: Processing:%d/%d | P1: Processing:%d/%d | P2: Processing:%d/%d | P3: Processing:%d/%d | P4: Processing:%d/%d | P5: Processing:%d/%d | P6: Processing:%d/%d | P7: Processing:%d/%d | P8: Processing:%d/%d | P9: Processing:%d/%d' %
                  (msg0, len(trainrecords[0]),
                   msg1, len(trainrecords[1]),
                   msg2, len(trainrecords[2]),
                   msg3, len(trainrecords[3]),
                   msg4, len(trainrecords[4]),
                   msg5, len(trainrecords[5]),
                   msg6, len(trainrecords[6]),
                   msg7, len(trainrecords[7]),
                   msg8, len(trainrecords[8]),
                   msg9, len(trainrecords[9])))
            time.sleep(10)
        except:
            raise Exception('')
