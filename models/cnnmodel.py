import tensorflow as tf
from abc import abstractmethod, ABC
from labeled_images.colormode import ColorMode


class CNNModel(ABC):
    def __init__(self, seed: int, lr: float, img_dim: int, color_mode: ColorMode = ColorMode.rgb):
        """ Creates layers for model and compiles model"""
        self.seed: int = seed
        self.lr: float = lr
        self.img_dim: int = img_dim
        self.model = None
        self.color = color_mode

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
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=0.00001,
                                                  decay=0.01,
                                                  amsgrad=False)
        self.model.compile(
            # optimizer='adam',
            optimizer=adam_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
