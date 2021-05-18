import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

SEED = 1

if __name__ == '__main__':
    # tf.random.set_seed(SEED)
    # (train_ds, validation_ds, test_ds), info = tfds.load(
    train_ds, validation_ds, test_ds = tfds.load(
            'emnist/balanced',
        split=['train[:40%]', 'train[40%:50%]', 'train[50%:60%]'],
        as_supervised=True  # , with_info=True
    )
    print('No of training samples: %d' % tf.data.experimental.cardinality(train_ds))
    print('No of validation samples: %d' % tf.data.experimental.cardinality(validation_ds))
    print('No of training samples: %d' % tf.data.experimental.cardinality(test_ds))

    label_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                   10: 'A', 11: 'B', 12: 'Cc', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'Ii',
                   19: 'Jj', 20: 'Kk', 21: 'L', 22: 'M', 23: 'Nn', 24: 'O', 25: 'P', 26: 'Qq', 27: 'Rr', 28: 'S',
                   29: 'Tt', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
                   36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n',
                   44: 'q', 45: 'r', 46: 't'}
    # reused: c, i, j, k, l, m, o, p, s, u, v, w, x, y, z
    # label_values = info.features['label'].names
    # plt.figure(figsize=(7, 7))
    # for i, (image, label) in enumerate(train_ds.take(25)):
    #     ax = plt.subplot(5, 5, i + 1)
    #     plt.imshow(image)
    #     plt.title(label_names[int(label)])
    #     plt.axis('off')

    # resize the images
    size = (71, 71)  # Xception requires color (3-ch), and a minimum size of 71x71x3
    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    # batch load - use caching & prefetching to optimize loading speed
    batch_size = 32
    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    # add data augmentation & visualize it on one image
    data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ])
    # for images, labels in train_ds.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = images[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(
    #             tf.expand_dims(first_image, 0), training=True
    #         )
    #         plt.imshow(augmented_image[0].numpy().astype("int32"))
    #         plt.title(int(labels[i]))
    #         plt.axis("off")

    # build model
    # todo: weights= can be None, 'imagenet' (default), or a specific local file location.
    # todo: 1) try imagenet as a base for ENISTv2
    # todo: 2) try to find a pretrained ENISTv2 model to train with Steyermark letters
    minimum_input_shape = (size[0], size[1], 3)
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet, norm values to [-1, 1].
        # Note that these weights are tuned to [-1,1], so scale the input.
        input_shape=minimum_input_shape,
        include_top=False,  # Do not include the ImageNet classifier at the top.
    )
    base_model.trainable = False
    # Create new model on top
    inputs = keras.Input(shape=minimum_input_shape)
    x = data_augmentation(inputs)  # Apply random data augmentation
    # at this point, x is shape None, size[0], size[1], 3  (batch size, rows, cols, channels)

    # normalize the inputs from [0,255] to [-1, 1]
    # Normalization calculates as outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([255/2] * 3)
    var = mean ** 2
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])

    # The base model contains batch norm layers, so we want to keep those in inference mode even once we unfreeze the
    # base model for fine-tuning.  The steps below make sure that base_model is running in inference mode.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)  # input (batch_size, row, col, ch) --> output (batch_size, ch)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout, 20% of inputs are set to 0 during training
                                      # (and remaining 80% are scaled up).  Only happens when training=True
    outputs = keras.layers.Dense(47)(x) # todo: test if this change works
    model = keras.Model(inputs, outputs, name="enist_balanced_model")  # inputs=a keras.Input object or a list of those objects
                                          # outputs=this uses the Functional API, so outputs is the chain of layers
    # Functional API output e.g. from docs:
    # x = keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    # outputs = keras.layers.Dense(5, activation=tf.nn.softmax)(x)
    # model = tf.keras.Model(input=inputs, output=outputs)

    model.summary()
    # keras.utils.plot_model(model, "model_graph_demo.png")

    # Train the top layer
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # todo: try this
        metrics=[keras.metrics.Accuracy()],  # todo: try this, or just 'accuracy'
    )
    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    model.save('emnist.model')

    # Then, fine tuning
    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.Accuracy()],
    )

    epochs = 5
    history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    model.save('emnist.model')

    # todo: try test data set
    test_scores = model.evaluate(test_ds, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])