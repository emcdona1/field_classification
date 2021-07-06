import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
from models.cnnmodel import CNNModel
from labeled_images.colormode import ColorMode


class CnnLstm(CNNModel):
    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255,
                                                                          input_shape=(self.img_dim, self.img_dim,
                                                                                       3 if self.color == ColorMode.rgb
                                                                                       else 1))
        self.model.add(norm_layer)
        self.add_convolutional_layers()
        self.add_lstm_layers()
        self.add_hidden_layers()
        self.add_output_layers()
        self.compile_model()

    def add_convolutional_layers(self):
        channels = 3 if self.color == ColorMode.rgb else 1

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
        self.model.add(layers.Dense(500, activation='linear'))
        self.model.add(layers.Dense(500, activation='relu'))

    def add_output_layers(self):
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation=tf.keras.activations.softmax))
        print(self.model.summary())
