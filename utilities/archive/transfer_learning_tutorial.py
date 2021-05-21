# from tutorial: https://keras.io/guides/transfer_learning/
# Note: This code doesn't work with TF2.0 -- it works with TF >= 2.4, unsure of the exact requirement
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def print_weights(this_layer):
    print('weights: %i' % len(this_layer.weights))
    print('trainable_weights: %i' % len(this_layer.trainable_weights))
    print('non_trainable_weights: %i' % len(this_layer.non_trainable_weights))


def basic_examples():
    layer = keras.layers.Dense(3)
    layer.build((None, 4))
    print_weights(layer)

    # only built-in this_layer with non-trainable weights
    layer = keras.layers.BatchNormalization()
    layer.build((None, 4))
    print_weights(layer)

    # Layers & models also feature a boolean attribute trainable
    layer = keras.layers.Dense(3)
    layer.build((None, 4))  # create the weights
    layer.trainable = False
    print_weights(layer)

    # example of non-trainable
    layer1 = keras.layers.Dense(3, activation='relu')
    layer2 = keras.layers.Dense(3, activation='sigmoid')
    model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])
    layer1.trainable = False
    initial_layer1_weights_values = layer1.get_weights()
    model.compile(optimizer='adam', loss='mse')
    model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
    final_layer1_weights_values = layer1.get_weights()  # these won't have changed!
    np.testing.assert_allclose(
        initial_layer1_weights_values[0], final_layer1_weights_values[0]
    )  # raises AssertionError if all values aren't equal (w/in a tolerance, default 1e-7)
    np.testing.assert_allclose(
        initial_layer1_weights_values[1], final_layer1_weights_values[1]
    )

    # If you set trainable = False on a model or on any layer that has sublayers,
    # all children layers become non-trainable as well.

    # Transfer Learning basic workflow - "a typical transfer learning workflow"
    base_model = keras.applications.Xception(
        weights='imagenet',
        input_shape=(150, 150, 3),
        include_top=False  # Don't include Imagenet classifier at the top (top="end" of the model)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(150, 150, 3))  # now add a trainable model on top
    x = base_model(inputs,
                   training=False)  # False makes sure 'base model is running in inference mode', which will be used later in fine-tuning
    x = keras.layers.GlobalAveragePooling2D()(
        x)  # 'convert features w/ shape = base_model.output_shape[1:] into vectors'
    outputs = keras.layers.Dense(1)(x)  # binary classification
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    model.fit("DATASET", epochs=20, callbacks=..., validation_data=...)

    # # More lightweight workflow:
    # Instantiate a base model and load pre-trained weights into it.
    # Run your new dataset through it and record the output of one (or several) layers from the base model. This is called feature extraction.
    # Use that output as input data for a new, smaller model.

    ## Fine tuning
    # Once the model has converged, fine tuning involves unfreezing all (or part) of the base model and retraining the
    # whole model with a very low learning rate.
    # Pro: Can give incremental improvements. Con: Can lead to overfitting.
    base_model.trainable = True  # Any time you change the trainability, you need to recompile the model
    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # very low learning rate
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()])
    # Once a model is compiled, its behaviour should be "frozen"
    model.fit("DATASET", epochs=10, callbacks=..., validation_data=...)

    # Notes about BatchNormalization, which is full of exceptions:
    # The 2 non-trainable weights track the mean and variance of the inputs
    # If you set batch_norm.trainable = False, the layer freezes the weights AND runs in inference mode. (Others don't do the latter.)
    # When you unfreeze a model which CONTAINS a BatchNormalization layer (for fine tuning), keep BatchNormalization
    # layer in inference mode by passing training=False when you call the base model.


def cat_dog_from_tutorial():
    # Note: If you have your own dataset, you'll probably want to use the utility
    # tf.keras.preprocessing.image_dataset_from_directory to generate similar labeled dataset objects from
    # a set of images on disk filed into class-specific folders.
    # tfds.disable_progress_bar()
    # tfds.enable_progress_bar()
    # view all available data sets: tfds.list_builders()
    train_ds, validation_ds, test_ds = tfds.load(
        'cats_vs_dogs',
        split=['train[:40%]', 'train[40%:50%]', 'train[50%:60%]'],
        as_supervised=True  # include the labels
    )
    print('No of training samples: %d' % tf.data.experimental.cardinality(train_ds))
    print('No of validation samples: %d' % tf.data.experimental.cardinality(validation_ds))
    print('No of training samples: %d' % tf.data.experimental.cardinality(test_ds))
    label_name = ['cat', 'dog']
    # view first 9 images in training set -- note that they all different dims
    # plt.figure(figsize=(10, 10))
    # for i, (image, label) in enumerate(train_ds.take(9)):
    #     ax = plt.subplot(3, 3, i + 1)
    #     plt.imshow(image)
    #     plt.title(label_name[label])
    #     plt.axis('off')

    # resize the images
    size = (150, 150)
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
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        # Note that these weights are tuned to [-1,1], so scale the input.
        input_shape=(150, 150, 3),
        include_top=False,  # Do not include the ImageNet classifier at the top.
    )
    base_model.trainable = False
    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # normalize the inputs from [0,255] to [-1, 1]
    # Normalization calculates as outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([255 / 2] * 3)
    var = mean ** 2
    x = norm_layer(x)
    norm_layer.set_weights([mean, var])

    # The base model contains batch norm layers, so we want to keep those in inference mode even once we unfreeze the
    # base model for fine-tuning.  The steps below make sure that base_model is running in inference mode.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    # Train the top layer
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )
    epochs = 10
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    model.save('cat_dog.model')

    # Then, fine tuning
    base_model.trainable = True
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    epochs = 5
    model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
    model.save('cat_dog.model')


def binary_classification_from_file_system():
    # from this tutorial: https://www.tensorflow.org/tutorials/load_data/images#load_using_keraspreprocessing
    data_dir = 'frullania_tmp'
    # Required if running in Colab:
    # import os
    # for f in os.walk('frullania'):
    #     curr_dir = f[0]
    #     sub_folders = f[1]
    #     contained_files = f[2]
    #     for folder in sub_folders:
    #         if '.ipynb_checkpoints' in folder:
    #             os.removedirs(os.path.join('frullania', folder))

    image_size = (71, 71)
    # NOTE: doesn't load in TIFs
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                         image_size=image_size,
                                                                         labels='inferred', seed=1,
                                                                         validation_split=0.3, subset='training')
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                                              image_size=image_size,
                                                                              labels='inferred', seed=1,
                                                                              validation_split=0.3, subset='validation')
    label_name = train_ds.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(label_name[labels[i]])
            plt.axis("off")
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)  # (# of images, image dimensions, channels)
        print(labels_batch.shape)  # (# of images, )
        break

    # below is an alternative way to do buffered prefetching
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # add data augmentation & visualize it on one image
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)  # [0, 1]

    num_classes = 2
    model = tf.keras.Sequential([
        data_augmentation,
        normalization_layer,
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=10
    )
