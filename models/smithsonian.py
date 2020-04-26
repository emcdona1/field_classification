from models.cnnmodel import CNNModel
import tensorflow as tf


class SmithsonianModel(CNNModel):
    """ Creates the model architecture as outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017). """

    def add_convolutional_layers(self):
        # Two sets of convolutional layers
        self.model.add(tf.keras.layers.Conv2D(10, (5, 5), input_shape=(
            self.img_dim, self.img_dim, 3)))  # todo: only works for color images
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(40, (5, 5)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation('relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Output shape: 40 x 61 x 61
        self.model.add(tf.keras.layers.Flatten())

    def add_hidden_layers(self):
        self.model.add(tf.keras.layers.Dropout(0.5, seed=self.seed))  # drop out 50% and then * 2 (same # of layers)
        self.model.add(tf.keras.layers.Dense(500, activation='linear'))
        self.model.add(tf.keras.layers.Dense(500, activation='relu'))

    def add_output_layers(self):
        self.model.add(tf.keras.layers.Dense(2, activation='linear'))
        self.model.add(tf.keras.layers.Dense(2, activation="softmax"))
        print(self.model.summary())
