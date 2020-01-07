from image_importer import ColorMode
import tensorflow as tf
from keras import regularizers


class SmithsonianModel:
    def __init__(self, img_size, color_mode, seed, lr):
        """ Creates layers for model and compiles model -- this complies as closely
        as possible to the model outlined in Schuettpelz, Frandsen, Dikow, Brown, et al. (2017).
        """
        self.img_size = img_size
        self.color = ColorMode.RGB if color_mode else ColorMode.BW
        self.seed = seed

        self.model = None
        self.model = tf.keras.models.Sequential()
        self.convolutional_layers()
        self.hidden_layers()
        self.compile_model(lr)

    def convolutional_layers(self):
        # Input shape = image height x image width x 3 (if color) or 1 (if b&w)

        # -------------First set of Convolutional Layers--------------
        # 1. Convolution Layer: 10 filters of 5px by 5px
        self.model.add(tf.keras.layers.Conv2D(10, (5, 5)))  # ,
        # input_shape=(self.img_size, self.img_size,
        #              3 if self.color == ColorMode.RGB else 1)))
        # Output shape: 10 x 252 x 252

        # 2. Batch Normalization: Normalizes previous layer to have mean near 0 and S.D. near 1
        self.model.add(tf.keras.layers.BatchNormalization())
        # Output shape: 10 x 252 x 252

        # 3. Activation Layer: ReLU uses the formula of f(x)= x if x>0 and 0 if x<=0
        # Apparently it's a pretty common one for CNN so we're going with the flow here
        self.model.add(tf.keras.layers.Activation("relu"))
        # Output shape: 10 x 252 x 252

        # 4. Pooling function: the paper didn't specify function, but it seems that the Mathematica default is Max
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output shape: 10 x 126 x 126

        # -------------Second set of Convolutional Layers--------------
        # 5. Convolution Layer: 40 filters of 5px by 5px
        self.model.add(tf.keras.layers.Conv2D(40, (5, 5)))
        # Output shape: 40 x 122 x 122

        # 6. Batch Normalization Layer
        self.model.add(tf.keras.layers.BatchNormalization())
        # Output shape: 40 x 122 x 122

        # 7. Activation Layer: Same as above
        self.model.add(tf.keras.layers.Activation("relu"))
        # Output shape: 40 x 122 x 122

        # 8. Pooling again will decrease "image shape" by half since stride = 2
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Output shape: 40 x 61 x 61

    def hidden_layers(self):
        # ----------Hidden layers-----------

        # 9. Flattening Layer: Make pooled layers (that look like stacks of grids) into one "column" to feed into ANN
        self.model.add(tf.keras.layers.Flatten())

        # 10. Dropout Layer: In Mathematica Dropout[] has a rate of dropping 50% of elements then * by 2 -- ours does not
        self.model.add(tf.keras.layers.Dropout(0.5, seed=self.seed))

        self.model.add(tf.keras.layers.Dense(500,
                                             activation="linear",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(tf.keras.layers.Dense(500,
                                             activation="relu",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))

        self.model.add(tf.keras.layers.Dropout(0.25, seed=self.seed))

        # The output layer with 2 neurons, for 2 classes
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="linear",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(tf.keras.layers.Dense(2,
                                             activation="softmax",
                                             activity_regularizer=regularizers.l2(0.01),
                                             kernel_regularizer=regularizers.l2(0.05)))

    def compile_model(self, learning_rate):
        opt = tf.keras.optimizers.Adam(lr=learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.00001,
                                       decay=0.0,
                                       amsgrad=False)
        self.model.compile(optimizer=opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
