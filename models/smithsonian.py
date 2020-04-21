from models.cnnmodel import CNNModel
import tensorflow as tf
from keras import regularizers


class SmithsonianModel(CNNModel):
    """ Creates the model architecture as outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017). """

    def add_convolutional_layers(self):
        # Two sets of convolutional layers
        self.model.add(tf.keras.layers.Conv2D(10, (5, 5), input_shape=(
            self.img_dim, self.img_dim, 3)))  # todo: only works for color images
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(tf.keras.layers.Conv2D(40, (5, 5)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Activation("relu"))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Output shape: 40 x 61 x 61
        self.model.add(tf.keras.layers.Flatten())

    def add_hidden_layers(self):
        # TODO: In Mathematica, Dropout[] has a rate of dropping 50% of elements then * by 2 -- ours does not.
        self.model.add(tf.keras.layers.Dropout(0.5, seed=self.seed))

        self.model.add(tf.keras.layers.Dense(500, activation="linear"))  # ,
        # activity_regularizer=regularizers.l2(0.01),
        # kernel_regularizer=regularizers.l2(0.05)))

        self.model.add(tf.keras.layers.Dense(500, activation="relu"))  # ,
        # activity_regularizer=regularizers.l2(0.01),
        # kernel_regularizer=regularizers.l2(0.05)))

        # self.model.add(tf.keras.layers.Dropout(0.25, seed=self.seed))  # TODO: Noting that this layer is not actually in the smithsonian!

    def add_output_layers(self):
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="linear",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="softmax",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
        print(self.model.summary())
