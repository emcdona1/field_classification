import tensorflow as tf
from abc import abstractmethod, ABC
from labeled_images import ColorMode
from typing import Union, Tuple


class CNNModel(ABC):
    def __init__(self, seed: int, learning_rate: float,
                 img_dim: Union[Tuple[int, int], int],
                 color_mode: ColorMode = ColorMode.rgb):
        """ Creates layers for model and compiles model"""
        self.seed: int = seed
        self.learning_rate: float = learning_rate
        self.img_dim: Tuple[int, int] = (img_dim, img_dim) if type(img_dim) == int else img_dim
        self.model = None
        self.color = color_mode

    def reset_model(self):
        self.model = None
        self.layer_setup()

    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255,
                                                                          input_shape=(self.img_dim[0], self.img_dim[1],
                                                                                       3 if self.color == ColorMode.rgb
                                                                                       else 1))
        self.model.add(norm_layer)
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
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=0.00001,
                                                  decay=0.01,
                                                  amsgrad=False)
        # Check loss to see if it is best
        self.model.compile(
            # optimizer='adam',
            optimizer=adam_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
