import tensorflow as tf
import os
import numpy as np
#tf.enable_eager_execution()
from sklearn.preprocessing import StandardScaler

print(tf.__version__)
class StockModel():
    def __init__(self,hidden_size,num_layer,input_width,input_height,check_dir,log_dir,data_file_list,batch_size):
        """
        初始化参数以及session重启
        :param hidden_size:
        :param num_layer:
        :param input_width:
        :param input_height:
        :param check_dir:
        :param log_dir:
        :param data_file_list:
        :param batch_size:
        """
        self.input_data_placeholder = tf.placeholder(shape=(batch_size, input_height, input_width), dtype=tf.float32, name="input_data")
        self.input_data_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_data_length')
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.input_width = input_width
        self.input_height = input_height
        self.labels = tf.placeholder(shape=(None,input_height),dtype=tf.float32)
        self.check_dir = check_dir
        self.batch_size = batch_size
        self.data_file_list = data_file_list
        self.build_network()
        self.get_data_iterator()
        self.log_dir = log_dir
        self.restore_session()


    def get_data_iterator(self):
        """
        数据初始化
        :return:
        """
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

        dataset = tf.data.TFRecordDataset(self.data_file_list)

        dataset = dataset.map(parser).repeat().batch(self.batch_size).shuffle(buffer_size=1000)
        iterator = dataset.make_one_shot_iterator()
        self.input_, self.label_ = iterator.get_next()
        self.input_ = tf.layers.batch_normalization(self.input_)

    def restore_session(self):
        """
        session重启
        :return:
        """
        if not os.path.exists(self.check_dir):
            os.makedirs(self.check_dir)
        self.saver = tf.train.Saver(max_to_keep=3)
        self.merged = tf.summary.merge_all()

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        check_point = tf.train.get_checkpoint_state(self.check_dir)
        if check_point and check_point.model_checkpoint_path:
            print("restor session")
            self.saver.restore(self.sess,check_point.model_checkpoint_path)
        self.train_writer = tf.summary.FileWriter(self.log_dir,self.sess.graph)


    def build_network(self):
        """
        构建网络
        :return:
        """
        encoder_cell = self._build_cell(self.hidden_size, self.num_layer)
        query_outputs, query_final_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.input_data_placeholder,
                                                             sequence_length=self.input_data_len, time_major=False,
                                                             dtype=tf.float32)
        net = tf.layers.Dense(1024)(query_outputs)
        net = tf.layers.Dropout(0.5)(net)
        net = tf.layers.Dense(1)(net)
        net = tf.reshape(net, (-1, self.input_height))
        self.loss = tf.losses.mean_squared_error(net, self.labels)
        tf.summary.scalar('loss', self.loss)
        self.result_out_put = net[:,-1]
        opt = tf.train.AdamOptimizer(
            learning_rate=0.0001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        #protect gradient explosion
        gvs = opt.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = opt.apply_gradients(capped_gvs)

    def train(self,max_iterator):
        index = 0
        for i in range(max_iterator):
            input_len = np.zeros((self.batch_size))

            img, lbl = self.sess.run([self.input_, self.label_])
            where_are = np.isnan(img)
            where_are_inf = np.isinf(img)
            img[where_are] = 0
            img[where_are_inf] = 0
            input_len[:] = 30
            index += 1
            summary,_, loss_, prediction = self.sess.run([self.merged,self.train_op, self.loss, self.result_out_put],
                                                 feed_dict={self.input_data_placeholder: img,
                                                            self.input_data_len: input_len, self.labels:lbl})
            self.train_writer.add_summary(summary,i)
            if i % 10==0:
                self.saver.save(self.sess, self.check_dir + "/model.ckpt")
                print("loss {}".format(loss_))
                print("prediction {}".format(prediction))
                labels = lbl
                lablelist = []
                for ele in labels:
                    lablelist.append(ele[-1])
                print("last predict {}".format(lablelist))
                # print("---")
                #print(img)
                # print("---")
                # print(lbl)
                # print("---")
                # print(input_len)
                # print("---")
        self.train_writer.close()
    def variable_summaries(self,var):
        """Attach a lot of summaries to a tensor"""
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean",mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar("stddev",stddev)
            tf.summary.scalar("max",tf.reduce_max(var))
            tf.summary.scalar("min",tf.reduce_min(var))
            tf.summary.histogram("histogram",var)

    @staticmethod
    def _build_cell(hidden_size,num_layer):
        encoder_cell = tf.contrib.rnn.MultiRNNCell(
            [StockModel.create_lstm(hidden_size) for _ in range(num_layer)]
        )
        return encoder_cell


    @staticmethod
    def create_lstm(hidden_size):
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
        return cell

if __name__ == '__main__':
    model = StockModel(512,1,15,30,'check_dir','log_dir','handle_data/stock_tf.tfrecode',1)
    model.train(1000000)