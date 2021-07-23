import tensorflow as tf
from tensorflow.keras import layers
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CnnLstm(CNNModel):
    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255,
                                                                          input_shape=(800, 64, 1))  # todo: no hardcode
        # self.img_dim, self.img_dim,
        # 3 if self.color == ColorMode.rgb
        # else 1))
        self.model.add(norm_layer)
        self.add_convolutional_layers()
        # self.add_rnn_layers() todo next
        self.model.summary()
        # self.add_hidden_layers() todo - probably get rid of abc requirement?
        # self.add_output_layers() todo - and rename to better match the functionality
        # self.compile_model()

    def add_convolutional_layers(self):
        """Create CNN layers."""

        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3, 3, 3]
        feature_vals = [64, 128, 128, 256, 512, 512]
        stride_vals = pool_vals = [(2, 2), (1, 2), (2, 2), (2, 2), (1, 2), (1, 2)]
        num_layers = len(stride_vals)  # 6

        # create layers
        for i in range(num_layers):
            print(i)
            if i == 3:
                self.model.add(tf.keras.layers.Conv2D(filters=256,
                                                      kernel_size=(kernel_vals[i], kernel_vals[i]),
                                                      padding='SAME',
                                                      activation='relu'))
            set_of_layers = tf.keras.Sequential()
            set_of_layers.add(
                tf.keras.layers.Conv2D(filters=feature_vals[i],
                                       kernel_size=(kernel_vals[i], kernel_vals[i]),
                                       padding='SAME',
                                       activation='relu'))
            set_of_layers.add(tf.keras.layers.BatchNormalization())
            set_of_layers.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_vals[i][0], pool_vals[i][1]),
                                                           strides=(stride_vals[i][0], stride_vals[i][1]),
                                                           padding='VALID'))
            self.model.add(set_of_layers)

    def add_rnn_layers(self):
        """Create RNN layers."""
        rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])

        # basic cells which is used to build RNN
        num_hidden = 256
        cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True) for _ in
                 range(2)]  # 2 layers
        # cells = [tf.keras.layers.LSTM(units=num_hidden), tf.keras.layers.LSTM(units=num_hidden)]

        # stack basic cells
        stacked = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        (fw, bw), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnn_in3d,
                                                                dtype=rnn_in3d.dtype)

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, num_hidden * 2, len(self.char_list) + 1], stddev=0.1))
        self.rnn_out_3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'),
                                     axis=[2])

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):  # CTC layers
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation=tf.keras.activations.softmax))
        print(self.model.summary())
