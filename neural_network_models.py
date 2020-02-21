from image_handling import ColorMode
import tensorflow as tf
from keras import regularizers
from abc import abstractmethod

class CNNModel:
    def __init__(self, img_size, color_mode, seed, lr):
        """ Creates layers for model and compiles model"""
        self.img_size = img_size
        self.color = ColorMode.RGB if color_mode else ColorMode.BW
        self.seed = seed
        self.lr = lr
        self.model = None
        self.layer_setup()

    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        self.add_convolutional_layers()
        self.add_hidden_layers()
        self.add_output_layers()
        self.compile_model()

    @abstractmethod
    def add_convolutional_layers(self):
        raise NotImplementedError()

    @abstractmethod
    def add_hidden_layers(self):
        raise NotImplementedError()

    @abstractmethod
    def add_output_layers(self):
        raise NotImplementedError()

    def compile_model(self):
        opt = tf.keras.optimizers.Adam(lr=self.lr,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.00001,
                                       decay=0.0,
                                       amsgrad=False)
        self.model.compile(optimizer=opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])


class SmithsonianModel(CNNModel):
    """ Creates the model architecture as outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017). """

    def convolutional_layers(self):
        # Input shape = image height x image width x 3 (if color) or 1 (if b&w)
        input_image_shape = (self.img_size, self.img_size, 3 if self.color == ColorMode.RGB else 1)

        # Two sets of convolutional layers
        for idx in range(0, 2):
            if idx == 0:
                self.model.add(tf.keras.layers.Conv2D(10, (5, 5)), input_shape=input_image_shape)
            else:
                self.model.add(tf.keras.layers.Conv2D(40, (5, 5)))
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation("relu"))
            self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Output shape: 40 x 61 x 61
        self.model.add(tf.keras.layers.Flatten())

    def hidden_layers(self):
        # TODO: In Mathematica, Dropout[] has a rate of dropping 50% of elements then * by 2 -- ours does not.
        self.model.add(tf.keras.layers.Dropout(0.5, seed=self.seed))
        
        self.model.add(tf.keras.layers.Dense(500, activation="linear",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))

        self.model.add(tf.keras.layers.Dense(500, activation="relu",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))

        self.model.add(tf.keras.layers.Dropout(0.25, seed=self.seed))

    def add_output_layers(self):
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="linear",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="softmax",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
