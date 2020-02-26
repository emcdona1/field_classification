from image_handling import ColorMode
import tensorflow as tf
from abc import abstractmethod, ABC


class CNNModel(ABC):
    def __init__(self, img_size, color_mode, seed, lr):
        """ Creates layers for model and compiles model"""
        self.img_size = img_size
        self.color = ColorMode.RGB if color_mode else ColorMode.BW
        self.seed = seed
        self.lr = lr
        self.model = None
        self.layer_setup()

    def reset_model(self):
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
