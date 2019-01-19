import tensorflow as tf

print(tf.__version__)
import numpy as np
from sklearn.preprocessing import StandardScaler



def read_tfrecord(file_name):
    input_data_placeholder = tf.placeholder(shape=(None, 30, 15), dtype=tf.float32,
                                            name="input_data")
    labels = tf.placeholder(shape=(None, 30), dtype=tf.float32)
    feature = {
        "stock_feature": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.string)
    }

    def parser(tf_record):
        features = tf.parse_single_example(tf_record, features=feature)
        data0 = tf.decode_raw(features['stock_feature'], tf.float32)
        data1 = tf.decode_raw(features['label'], tf.float32)

        stock_feature = tf.reshape(data0, [30, 15])
        close = tf.reshape(data1, [30])
        return stock_feature, close

    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(parser).repeat().batch(2).shuffle(buffer_size=1000)
    iterator = dataset.make_one_shot_iterator()

    input_,label_ = iterator.get_next()
    print(input_,label_)
    print(input_data_placeholder,labels)
    scaler = StandardScaler()

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        for batch_index in range(1):
            img, lbl = sess.run([input_, label_])
            where_are = np.isnan(img)
            where_are_inf = np.isinf(img)
            img[where_are] = 0
            img[where_are_inf] = 0
            print(img)

if __name__ == '__main__':
    filename = "stock_tf.tfrecode"
    read_tfrecord(filename)