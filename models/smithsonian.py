from models.cnnmodel import CNNModel
from tensorflow.keras import layers
from labeled_images.colormode import ColorMode


class SmithsonianModel(CNNModel):
    """ Creates the model architecture as outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017). """

    def add_convolutional_layers(self):
        # Two sets of convolutional layers
        if self.color == ColorMode.RGB:
            print('Color image input layer')
            self.model.add(layers.Conv2D(10, (5, 5), input_shape=(
                self.img_dim, self.img_dim, 3)))
        else:
            print('Grayscale image input layer')
            self.model.add(layers.Conv2D(10, (5, 5), input_shape=(
                self.img_dim, self.img_dim, 1)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(layers.Conv2D(40, (5, 5)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Activation('relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Output shape: 40 x 61 x 61
        self.model.add(layers.Flatten())

    def add_hidden_layers(self):
        self.model.add(layers.Dropout(0.5, seed=self.seed))  # drop out 50% and then * 2 (same # of layers)
        self.model.add(layers.Dense(500, activation='linear'))
        self.model.add(layers.Dense(500, activation='relu'))

    def add_output_layers(self):
        self.model.add(layers.Dense(2, activation='linear'))
        self.model.add(layers.Dense(2, activation='softmax'))
        print(self.model.summary())
