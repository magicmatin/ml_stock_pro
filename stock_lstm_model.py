import tensorflow as tf


class StockModel():
    def __init__(self,hidden_size,num_layer,input_width,input_height,episole):
        self.input_data_placeholder = tf.placeholder(shape=(None, input_height, input_width), dtype=tf.float32, name="input_data")
        self.input_data_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_data_length')
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.input_width = input_width
        self.input_height = input_height
        self.labels = tf.placeholder(shape=(None,input_height),dtype=tf.float32)
        self.episole = episole
        self.build_network()


    def build_network(self):
        encoder_cell = self._build_cell(self.hidden_size, self.num_layer)
        query_outputs, query_final_state = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=self.input_data_placeholder,
                                                             sequence_length=self.input_data_len, time_major=False,
                                                             dtype=tf.float32)
        net = tf.layers.Dense(1024)(query_outputs)
        net = tf.layers.Dropout(0.5)(net)
        net = tf.layers.Dense(1)(net)
        net = tf.reshape(net, (-1, self.input_height))

        self.loss = tf.losses.mean_squared_error(net, self.labels)
        self.result_out_put = net[:,-1]
        opt = tf.train.AdamOptimizer(
            learning_rate=0.0001,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08
        )
        self.optOp = opt.minimize(self.loss)

    def train(self,batches):
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        for index in range(self.episole):
            for b in batches:
                _, loss_,prediction = sess.run([self.optOp, self.loss,self.result_out_put],
                                             feed_dict={self.input_data_placeholder: b.input_data,
                                                        self.input_data_len: b.input_len, self.labels: b.label})
                if index % 100:
                    print(loss_)
                    print(prediction)
                    labels = b.label
                    lablelist = []
                    for ele in labels:
                        lablelist.append(ele[-1])
                    print(lablelist)

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
