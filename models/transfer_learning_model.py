import tensorflow as tf
from labeled_images import ColorMode
from models import CNNModel


class TransferLearningModel(CNNModel):
    def __init__(self, base_model: tf.keras.models.Model, seed: int, learning_rate: float, img_dim: int,
                 color_mode: ColorMode = ColorMode.rgb):
        """ Creates layers for model and compiles model"""
        super().__init__(seed, learning_rate, img_dim, color_mode)
        self.base_model = base_model

    def layer_setup(self):
        self.model = tf.keras.models.Sequential()
        norm_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1/255,
                                                                          input_shape=(self.img_dim, self.img_dim,
                                                                                       3 if self.color == ColorMode.rgb
                                                                                       else 1))
        self.model.add(norm_layer)
        self.add_convolutional_layers()
        self.add_hidden_layers()
        self.add_output_layers()
        self.compile_model()

    def add_convolutional_layers(self):
        pass

    def add_hidden_layers(self):
        pass

    def add_output_layers(self):
        raise NotImplementedError()

    def compile_model(self):
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  epsilon=0.00001,
                                                  decay=0.01,
                                                  amsgrad=False)
        self.model.compile(
            optimizer=adam_optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
