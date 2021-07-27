# Resources:
# https://github.com/githubharald/SimpleHTR
# https://stackoverflow.com/questions/63258412/adding-ctc-loss-and-ctc-decode-to-a-keras-model#63306211
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CnnLstm(CNNModel):
    def layer_setup(self):
        self.inputs = keras.Input(shape=(800, 64, 1))  # todo: not hard-coding the shape? or preprocess?
        self.model = keras.layers.experimental.preprocessing.Rescaling(1 / 255)(self.inputs)
        self.add_convolutional_layers()
        self.add_rnn_layers()
        self.add_output_layers()
        self.outputs = self.model
        self.testing_model = keras.Model(inputs=self.inputs, outputs=self.outputs, name='testing_model')
        self.testing_model.summary()
        self.model = None
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
                self.model = layers.Conv2D(filters=256,
                                                      kernel_size=(kernel_values[i], kernel_values[i]),
                                                      padding='SAME',
                                                      activation='relu')(self.model)
            set_of_layers = keras.Sequential()
            set_of_layers.add(
                layers.Conv2D(filters=feature_values[i],
                                       kernel_size=(kernel_values[i], kernel_values[i]),
                                       padding='SAME',
                                       activation='relu'))
            set_of_layers.add(layers.BatchNormalization())
            set_of_layers.add(layers.MaxPooling2D(pool_size=(pool_values[i][0], pool_values[i][1]),
                                                           strides=(stride_values[i][0], stride_values[i][1]),
                                                           padding='VALID'))
            self.model = set_of_layers(self.model)

    def add_rnn_layers(self):
        """Create RNN layers."""
        self.model = layers.Lambda(lambda t: t[:, :, 0, :])(self.model)
        # rnn_in3d = tf.squeeze(self.cnn_out_4d, axis=[2])  # todo

        # basic cells which is used to build RNN
        num_hidden = 512

        rnn = [layers.LSTMCell(num_hidden), layers.LSTMCell(num_hidden)]
        rnn = layers.StackedRNNCells(rnn)
        rnn = layers.RNN(rnn, return_sequences=True)
        self.model = layers.Bidirectional(rnn)(self.model)

    def add_hidden_layers(self):
        # BxTxC -> TxBxC
        self.ctc_in_3d_tbc = tf.transpose(a=self.rnn_out_3d, perm=[1, 0, 2])
        # ground truth text as sparse tensor
        self.gt_texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape=[None, 2]),
                                        tf.compat.v1.placeholder(tf.int32, [None]),
                                        tf.compat.v1.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seq_len = tf.compat.v1.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(
            input_tensor=tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.ctc_in_3d_tbc,
                                                  sequence_length=self.seq_len,
                                                  ctc_merge_repeated=True))

        # calc loss for each element to compute label probability
        self.saved_ctc_input = tf.compat.v1.placeholder(tf.float32,
                                                        shape=[None, None, len(self.char_list) + 1])
        self.loss_per_element = tf.compat.v1.nn.ctc_loss(labels=self.gt_texts, inputs=self.saved_ctc_input,
                                                         sequence_length=self.seq_len, ctc_merge_repeated=True)

        # best path decoding or beam search decoding
        self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctc_in_3d_tbc, sequence_length=self.seq_len)

    def add_output_layers(self):
        sequence_length = len(self.char_list) + 1
        self.model = layers.Dense(sequence_length, activation='softmax')(self.model)
