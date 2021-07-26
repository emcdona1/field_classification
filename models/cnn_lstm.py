# Resources:
# https://github.com/githubharald/SimpleHTR
# https://stackoverflow.com/questions/63258412/adding-ctc-loss-and-ctc-decode-to-a-keras-model#63306211
import tensorflow as tf
from tensorflow.keras import layers
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CnnLstm(CNNModel):
    def layer_setup(self):
        model_input = tf.keras.Input(shape=(800, 64, 1))  # todo: not hard-coding the shape? or preprocess?
        self.model = tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255)(model_input)
        self.add_convolutional_layers()
        self.add_rnn_layers()
        # self.add_hidden_layers()  # actually CTC
        output = tf.keras.layers.Dense(10)(self.model)
        model_tmp = tf.keras.Model(inputs=model_input, outputs=output, name='tmp')
        model_tmp.summary()
        # self.add_output_layers() todo - and rename to better match the functionality
        # self.compile_model()

    def add_convolutional_layers(self):
        """Create CNN layers."""

        # list of parameters for the layers
        kernel_values = [5, 5, 3, 3, 3, 3]
        feature_values = [64, 128, 128, 256, 512, 512]
        stride_values = pool_values = [(2, 2), (1, 2), (2, 2), (2, 2), (1, 2), (1, 2)]
        num_layers = len(stride_values)  # 6

        # create layers
        for i in range(num_layers):
            if i == 3:
                self.model = tf.keras.layers.Conv2D(filters=256,
                                                      kernel_size=(kernel_values[i], kernel_values[i]),
                                                      padding='SAME',
                                                      activation='relu')(self.model)
            set_of_layers = tf.keras.Sequential()
            set_of_layers.add(
                tf.keras.layers.Conv2D(filters=feature_values[i],
                                       kernel_size=(kernel_values[i], kernel_values[i]),
                                       padding='SAME',
                                       activation='relu'))
            set_of_layers.add(tf.keras.layers.BatchNormalization())
            set_of_layers.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_values[i][0], pool_values[i][1]),
                                                           strides=(stride_values[i][0], stride_values[i][1]),
                                                           padding='VALID'))
            self.model = set_of_layers(self.model)

    def add_rnn_layers(self):
        """Create RNN layers."""
        self.model = tf.keras.layers.Reshape((100, 512), input_shape=(100, 1, 512))(self.model)
        # rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])  # todo

        # basic cells which is used to build RNN
        num_hidden = 512

        rnn = [tf.keras.layers.LSTMCell(num_hidden), tf.keras.layers.LSTMCell(num_hidden)]
        rnn = tf.keras.layers.StackedRNNCells(rnn)
        rnn = tf.keras.layers.RNN(rnn, return_sequences=True)
        self.model = tf.keras.layers.Bidirectional(rnn)(self.model)

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):  # CTC layers
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation=tf.keras.activations.softmax))
        print(self.model.summary())
