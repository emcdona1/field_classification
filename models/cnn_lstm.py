# Resources:
# https://github.com/githubharald/SimpleHTR
# https://stackoverflow.com/questions/63258412/adding-ctc-loss-and-ctc-decode-to-a-keras-model#63306211
# https://keras.io/examples/vision/captcha_ocr/
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)

    def call(self, y_true, y_pred):  # todo: figure out why this is flagged but works fine?
        # Compute the training time loss & add it to the layer  using self.add_loss()
        batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_len = tf.cast(tf.shape(y_pred)[0], dtype='int64')
        label_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
        input_len = input_len * tf.ones(shape=(batch_len, 1), dtype='int64')
        label_len = label_len * tf.ones(shape=(batch_len, 1), dtype='int64')

        loss = K.ctc_batch_cost(y_true, y_pred, input_len, label_len)
        self.add_loss(loss)
        return y_pred


class CnnLstm(CNNModel):
    def __init__(self, seed: int, learning_rate: float, img_dim: int, color_mode: ColorMode = ColorMode.rgb):
        super().__init__(seed, learning_rate, img_dim, color_mode)
        self.char_list: str = '\' !"#&()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.num_labels = len(self.char_list) + 1
        self.inputs = None
        self.inputs_labels = None
        self.model: keras.models.Model = None  # training model

    def layer_setup(self):
        self.inputs = keras.Input(shape=(800, 64, 1), name='image_in', dtype='float32')
        # todo: not hard-coding the shape? or preprocess?
        self.model = keras.layers.experimental.preprocessing.Rescaling(1 / 255)(self.inputs)
        self.add_convolutional_layers()
        self.add_rnn_layers()
        self.add_output_layers()
        self.model = keras.Model(inputs=[self.inputs, self.inputs_labels],
                                 outputs=self.model,
                                 name='ctc_training_model')
        self.compile_model()

    def add_convolutional_layers(self):
        """Create CNN layers."""

        # list of parameters for the layers
        kernel_values = [5, 5, 3, 3, 3, 3]
        feature_values = [64, 128, 128, 256, 512, 512]
        stride_values = pool_values = [(2, 2), (1, 2), (2, 2), (2, 2), (1, 2), (1, 2)]
        num_layers = len(stride_values)  # 6

        # create layers
        for i in range(num_layers):  # todo: get rid of this for loop because it's hard to read / dumb
            if i == 3:
                self.model = layers.Conv2D(name='stacked_conv',
                                           filters=256,
                                           kernel_size=(kernel_values[i], kernel_values[i]),
                                           padding='SAME',
                                           activation='relu')(self.model)
            set_of_layers = keras.Sequential(name=f'conv_block_{i + 1}')
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
        self.model = layers.Lambda(lambda t: t[:, :, 0, :], name='squeeze_middle_dim')(self.model)

        # basic cells which is used to build RNN
        num_hidden = 512
        rnn = [layers.LSTMCell(num_hidden), layers.LSTMCell(num_hidden)]
        rnn = layers.StackedRNNCells(rnn)
        rnn = layers.RNN(rnn, return_sequences=True)
        self.model = layers.Bidirectional(rnn, name='bidirectional_stacked_lstm')(self.model)

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):
        self.inputs_labels = layers.Input(name='labels', shape=[self.num_labels],
                                          dtype='float32')  # all characters + CTC 'blank'

        self.model = layers.Dense(self.num_labels, activation='relu')(self.model)
        self.model = layers.Dense(self.num_labels, name='dense_labels', activation='softmax')(self.model)
        self.model = CTCLayer(name='ctc_loss')(self.inputs_labels, self.model)

    def compile_model(self):
        rms_optimizer = tf.keras.optimizers.RMSprop()
        self.model.compile(
            # optimizer='adam',
            optimizer=rms_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def get_inference_model(self) -> tf.keras.Model:
        return keras.Model(inputs=self.inputs,
                           outputs=self.model.get_layer(name='dense_labels').output,
                           name='inference_model')
