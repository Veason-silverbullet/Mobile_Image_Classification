import os
import sys
import urllib
import tarfile
import shutil
import tensorflow as tf
import Inception
import ResNet
import Xception
import MobileNet
import ResNeXt
import ShuffleNet

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 25, """Batch size""")
tf.app.flags.DEFINE_integer('epoch', 300, """Training epoch num""")
tf.app.flags.DEFINE_integer('train_image_num', 50000, """Num of training images""")
tf.app.flags.DEFINE_integer('test_image_num', 10000, """Num of testing images""")
tf.app.flags.DEFINE_string('file_name_list', 'file_name_list.txt', """Training data directory""")
tf.app.flags.DEFINE_string('train_dir', 'train', """Training data directory""")
tf.app.flags.DEFINE_string('eval_dir', 'eval', """Evaluation data directory""")
tf.app.flags.DEFINE_string('test_dir', 'test', """Testing data directory""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt', """Model check point directory""")
tf.app.flags.DEFINE_string('summary_dir', './summary', """Directory to log summary""")
tf.app.flags.DEFINE_integer('step_to_show_result', 100, """Steps to show result""")
tf.app.flags.DEFINE_bool('log_device_placement', True, 'Log device placement')

tf.app.flags.DEFINE_string('dataset', 'ImageNet', """Dataset to train""")
tf.app.flags.DEFINE_string('network', 'ShuffleNet-v2', """Network""")
tf.app.flags.DEFINE_string('train_label_file', 'train_label.txt', """Train label file""")
tf.app.flags.DEFINE_string('validation_label_file', 'validation_label.txt', """Validation label file""")

tf.app.flags.DEFINE_string('optimizer', 'sgd', """Optimizer""")
tf.app.flags.DEFINE_string('lr_policy', 'exponential_decay', """Learning rate decay policy""")
tf.app.flags.DEFINE_float('init_learning_rate', 0.045, """Initial learning rate""")
tf.app.flags.DEFINE_float('end_learning_rate', 0, """End learning rate""")
tf.app.flags.DEFINE_float('lr_decay_power', 1, """Learning rate decay power""")
tf.app.flags.DEFINE_float('momentum', 0.9, """Momentum""")
tf.app.flags.DEFINE_bool('useNAG', True, """Whether use NAG""")
tf.app.flags.DEFINE_bool('gradient_clipping', True, """Whether clip gradient""")
tf.app.flags.DEFINE_float('clipping_gradient', 2, """Clipping gradient threshold""")
tf.app.flags.DEFINE_integer('step_to_save_model', 10000, """Steps to save the model""")
tf.app.flags.DEFINE_float('dropout_rate', 0.2, """Drop out rate""")
tf.app.flags.DEFINE_float('L2Reg_weight', 0.00004, """L2 regulation weight""")


if not os.path.exists(args.train_dir):
    os.mkdir(args.train_dir)
if not os.path.exists(args.eval_dir):
    os.mkdir(args.eval_dir)
if not os.path.exists(args.test_dir):
    os.mkdir(args.test_dir)
if os.path.exists(args.train_dir + '/train.bin') is False or os.path.exists(args.test_dir + '/test.bin') is False:
    data_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

    def progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    raw_data_path, _ = urllib.request.urlretrieve(data_url, args.train_dir + '/data', progress)
    data_path = raw_data_path + '~'
    if not os.path.exists(data_path):
        tarfile.open(raw_data_path, 'r:gz').extractall(args.train_dir)
    shutil.move(args.train_dir + '/cifar-100-binary/train.bin', args.train_dir + '/train.bin')
    shutil.move(args.train_dir + '/cifar-100-binary/test.bin', args.test_dir + '/test.bin')


def main():
    if args.dataset == 'Cifar-100' or args.dataset == 'cifar-100':
        label_dim = 100
    elif args.dataset == 'ImageNet' or args.dataset == 'imageNet' or args.dataset == 'imagenet':
        label_dim = 1000
    else:
        raise Exception('Unexpected dataset \"%s\". Dataset must be [ImageNet | Cifar-100]', args.dataset)

    if args.network == 'Inception_V3' or args.network == 'inception_V3' or args.network == 'Inception_v3' or args.network == 'inception_v3':
        inception_v3 = Inception.Inception_V3(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        inception_v3.train()
    elif args.network == 'ResNet_18' or args.network == 'resNet_18':
        resNet_18 = ResNet.ResNet_18(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNet_18.train()
    elif args.network == 'ResNet_34' or args.network == 'resNet_34':
        resNet_34 = ResNet.ResNet_34(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNet_34.train()
    elif args.network == 'ResNet_50' or args.network == 'resNet_50':
        resNet_50 = ResNet.ResNet_50(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNet_50.train()
    elif args.network == 'ResNet_101' or args.network == 'resNet_101':
        resNet_101 = ResNet.ResNet_101(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNet_101.train()
    elif args.network == 'ResNet_152' or args.network == 'resNet_152':
        resNet_152 = ResNet.ResNet_152(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNet_152.train()
    elif args.network == 'Xception' or args.network == 'xception':
        xception = Xception.Xception(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        xception.train()
    elif args.network == 'MobileNet_V1' or args.network == 'mobileNet_V1' or args.network == 'MobileNet_v1' or args.network == 'mobileNet_v1':
        mobileNet_v1 = MobileNet.MobileNet_V1(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        mobileNet_v1.train()
    elif args.network == 'MobileNet_V2' or args.network == 'mobileNet_V2' or args.network == 'MobileNet_v2' or args.network == 'mobileNet_v2':
        mobileNet_v2 = MobileNet.MobileNet_V2(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        mobileNet_v2.train()
    elif args.network == 'ResNeXt_50' or args.network == 'resNeXt_50':
        resNeXt_50 = ResNeXt.ResNeXt_50(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNeXt_50.train()
    elif args.network == 'ResNeXt_101' or args.network == 'resNeXt_101':
        resNeXt_101 = ResNeXt.ResNeXt_101(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNeXt_101.train()
    elif args.network == 'ResNeXt_152' or args.network == 'resNeXt_152':
        resNeXt_152 = ResNeXt.ResNeXt_152(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        resNeXt_152.train()
    elif args.network == 'ShuffleNet_V1' or args.network == 'shuffleNet_V1' or args.network == 'ShuffleNet_v1' or args.network == 'shuffleNet_v1':
        shuffleNet_v1 = ShuffleNet.ShuffleNet_V1(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        shuffleNet_v1.train()
    elif args.network == 'ShuffleNet_V2' or args.network == 'shuffleNet_V2' or args.network == 'ShuffleNet_v2' or args.network == 'shuffleNet_v2':
        shuffleNet_v2 = ShuffleNet.ShuffleNet_V2(args, tf.float32, label_dim=label_dim, initializer=tf.zeros_initializer)
        shuffleNet_v2.train()
    else:
        raise Exception('Unexpected network \"%s\".', args.network)


if __name__ == '__main__':
    tf.app.run()
