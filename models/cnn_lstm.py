import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CnnLstm(CNNModel):
    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255,
                                                                          input_shape=(800, 64, 1))
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

        # Two sets of convolutional layers
        self.model.add(TimeDistributed(layers.Conv2D(10, (5, 5), input_shape=(self.img_dim, self.img_dim, channels))))
        self.model.add(TimeDistributed(layers.BatchNormalization()))
        self.model.add(TimeDistributed(layers.Activation('relu')))
        self.model.add(TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

        self.model.add(TimeDistributed(layers.Conv2D(40, (5, 5))))
        self.model.add(TimeDistributed(layers.BatchNormalization()))
        self.model.add(TimeDistributed(layers.Activation('relu')))
        self.model.add(TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))

        # Output shape: 40 x 61 x 61
        self.model.add(TimeDistributed(layers.Flatten()))
        self.model.add(TimeDistributed(layers.Dropout(0.5, seed=self.seed)))
        # drop out 50% and then * 2 (same # of layers)

    def add_lstm_layers(self):
        self.model.add(layers.LSTM(5, input_shape=()))  # todo: huh?

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):  # CTC layers
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation=tf.keras.activations.softmax))
        print(self.model.summary())
