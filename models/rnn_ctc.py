# Resources:
# https://github.com/githubharald/SimpleHTR
# https://stackoverflow.com/questions/63258412/adding-ctc-loss-and-ctc-decode-to-a-keras-model#63306211
# https://keras.io/examples/vision/captcha_ocr/
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model, Sequential
from tensorflow.keras import backend as K
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode
from typing import Tuple

CHAR_LIST: str = '\' !"#&()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Compute the training time loss & add it to the layer  using self.add_loss()
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
        label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


class RnnCtc(CNNModel):
    def __init__(self, seed: int, learning_rate: float, img_dim: Tuple[int, int],
                 color_mode: ColorMode = ColorMode.grayscale):
        super().__init__(seed, learning_rate, img_dim, color_mode)
        self.num_labels = len(set(CHAR_LIST)) + 1
        self.inputs = None
        self.inputs_labels = None
        self.model = None  # training model

    def layer_setup(self):
        # Schiedl steps:
        # input_shape = (self.img_dim[1], self.img_dim[0], 3 if self.color == ColorMode.rgb else 1)
        # self.inputs = layers.Input(shape=input_shape, name='image', dtype='float32')
        # self.model = self.inputs
        # # self.model = layers.experimental.preprocessing.Rescaling(1 / 255)(self.inputs)
        # self.add_convolutional_layers()
        # self.add_rnn_layers()
        # output = self.add_output_layers()
        # self.model = Model(inputs=[self.inputs, self.inputs_labels],
        #                          outputs=output,
        #                          name='ctc_training_model')
        # self.compile_model()

        # Captcha steps
        self.inputs = layers.Input(shape=(self.img_dim[1], self.img_dim[0], 1), name='image', dtype='float32')
        self.model = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                                   name='Conv1', )(self.inputs)
        self.add_convolutional_layers()
        self.add_rnn_layers()
        output = self.add_output_layers()
        self.model = Model(inputs=[self.inputs, self.inputs_labels], outputs=output, name='ocr_model_v1')
        self.compile_model()

    def add_convolutional_layers(self):
        """Create CNN layers."""

        # Captcha steps
        # First conv block
        self.model = layers.MaxPooling2D((2, 2), name='pool1')(self.model)

        # Second conv block
        self.model = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same',
                                   name='Conv2')(self.model)
        self.model = layers.MaxPooling2D((2, 2), name='pool2')(self.model)

        # We have used two max pool with pool size and strides 2.
        # Hence, down-sampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = ((self.img_dim[1] // 4), (self.img_dim[0] // 4) * 64)
        self.model = layers.Reshape(target_shape=new_shape, name='reshape')(self.model)
        self.model = layers.Dense(64, activation='relu', name='dense1')(self.model)
        self.model = layers.Dropout(0.2)(self.model)

        # Scheidl steps
        # # list of parameters for the layers
        # kernel_values = [5, 5, 3, 3, 3, 3]
        # feature_values = [64, 128, 128, 256, 512, 512]
        # stride_values = pool_values = [(2, 2), (1, 2), (2, 2), (2, 2), (1, 2), (1, 2)]
        # num_layers = len(stride_values)  # 6
        #
        # # create layers
        # for i in range(num_layers):  # todo: get rid of this for loop because it's hard to read / dumb
        #     if i == 3:
        #         self.model = layers.Conv2D(name='stacked_conv',
        #                                    filters=256,
        #                                    kernel_size=(kernel_values[i], kernel_values[i]),
        #                                    padding='SAME',
        #                                    activation='relu')(self.model)
        #     set_of_layers = Sequential(name=f'conv_block_{i + 1}')
        #     set_of_layers.add(
        #         layers.Conv2D(filters=feature_values[i],
        #                       kernel_size=(kernel_values[i], kernel_values[i]),
        #                       padding='SAME',
        #                       activation='relu'))
        #     set_of_layers.add(layers.BatchNormalization())
        #     set_of_layers.add(layers.MaxPooling2D(pool_size=(pool_values[i][0], pool_values[i][1]),
        #                                           strides=(stride_values[i][0], stride_values[i][1]),
        #                                           padding='VALID'))
        #     self.model = set_of_layers(self.model)

    def add_rnn_layers(self):
        """Create RNN layers."""
        # Captcha steps
        self.model = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(self.model)
        self.model = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(self.model)

        # Scheidl steps
        # self.model = layers.Lambda(lambda t: t[:, :, 0, :], name='squeeze_middle_dim')(self.model)
        #
        # # basic cells which is used to build RNN
        # num_hidden = 512
        # rnn = [layers.LSTMCell(num_hidden), layers.LSTMCell(num_hidden)]
        # rnn = layers.StackedRNNCells(rnn)
        # rnn = layers.RNN(rnn, return_sequences=True)
        # self.model = layers.Bidirectional(rnn, name='bidirectional_stacked_lstm')(self.model)

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):
        # Captcha steps
        self.inputs_labels = layers.Input(name='label', shape=(None,), dtype='float32')
        self.model = layers.Dense(len(CHAR_LIST) + 2, activation='softmax', name='dense_labels')(self.model)
        return CTCLayer(name='ctc_loss')(self.inputs_labels, self.model)

        # Scheidl steps
        # self.inputs_labels = layers.Input(name='label', shape=(None, ),  # shape=[self.num_labels],
        #                                   dtype='float32')  # all characters + CTC 'blank'
        #
        # self.model = layers.Dense(self.num_labels, activation='relu')(self.model)
        # self.model = layers.Dense(self.num_labels, name='dense_labels', activation='softmax')(self.model)
        # return CTCLayer(name='ctc_loss')(self.inputs_labels, self.model)

    def compile_model(self):
        # Captcha steps
        opt = optimizers.Adam()
        self.model.compile(optimizer=opt)

        # Scheidl steps
        # rms_optimizer = optimizers.RMSprop()
        # self.model.compile(
        #     # optimizer='adam',
        #     optimizer=rms_optimizer,
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy'])

    def get_inference_model(self) -> Model:
        return Model(inputs=self.inputs,
                     outputs=self.model.get_layer(name='dense_labels').output,
                     name='inference_model')
