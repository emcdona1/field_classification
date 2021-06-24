from models import CNNModel
from tensorflow.keras import layers
import tensorflow as tf
from labeled_images.colormode import ColorMode


class SmithsonianModel(CNNModel):
    """ Creates the model architecture as outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017). """

    def add_convolutional_layers(self):
        channels = 3 if self.color == ColorMode.rgb else 1

        # Two sets of convolutional layers
        self.model.add(layers.Conv2D(10, (5, 5), input_shape=(self.img_dim, self.img_dim, channels)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(layers.Conv2D(40, (5, 5)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Output shape: 40 x 61 x 61
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(0.5, seed=self.seed))  # drop out 50% and then * 2 (same # of layers)

    def add_hidden_layers(self):
        self.model.add(layers.Dense(500, activation='linear'))
        self.model.add(layers.Dense(500, activation='relu'))

    def add_output_layers(self):
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation=tf.keras.activations.softmax))
        print(self.model.summary())
