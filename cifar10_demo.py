import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def main():
    # CIFAR, reduced -- color images in 2 classes, 10k train, 2k test
    (train_features, train_labels), (test_features, test_labels) = image_set()
    NUM_CLASSES = 2
    IMAGE_SHAPE = (32, 32, 3)
    SEED = 1

    # plot_sample_images(train_features, train_labels)

    model = models.Sequential()
    # convolutional layers
    model.add(layers.Conv2D(10, (5, 5), input_shape=IMAGE_SHAPE))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(40, (5, 5)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # hidden layers
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5, seed=SEED))

    model.add(layers.Dense(500, activation="linear"))  # ,
    # activity_regularizer=regularizers.l2(0.01),
    # kernel_regularizer=regularizers.l2(0.05)))
    # noting that these are imported from tf.keras, not just keras

    model.add(layers.Dense(500, activation="relu"))  # ,
    # activity_regularizer=regularizers.l2(0.01),
    # kernel_regularizer=regularizers.l2(0.05)))

    # model.add(layers.Dropout(0.25, seed=SEED))

    # output layer
    # model.add(layers.Dense(NUM_CLASSES))

    model.add(layers.Dense(2, activation="linear"))  # ,
    # activity_regularizer=regularizers.l2(0.01),
    # kernel_regularizer=regularizers.l2(0.05)))
    model.add(layers.Dense(2, activation="softmax"))  # ,
    # activity_regularizer=regularizers.l2(0.01),
    # kernel_regularizer=regularizers.l2(0.05)))

    print(model.summary())
    # compile & train
    adam_opt = tf.keras.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9,
                                        beta_2=0.999,
                                        epsilon=0.00001,
                                        decay=0.0,
                                        amsgrad=False)
    model.compile(optimizer=adam_opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_features, train_labels,
                        validation_data=(test_features, test_labels),
                        epochs=50,
                        batch_size=64,
                        verbose=2)

    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.49, 1.01])
    plt.legend(loc='lower right')
    plt.savefig('cifar10-cat-dog_acc_' + datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S') + '.png')

    test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
    print(test_acc)


def plot_sample_images(train_features, train_labels):
    class_names = ['cat', 'dog']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_features[i])
        # class labels are arrays, so you need to specify image as first dim, then 0
        print(train_labels[i][0])
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


def image_set():
    # CIFAR10 - color images in 10 classes, 50k train, 10k test
    (train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()
    train_features, test_features = train_features / 255.0, test_features / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # reduce to just cat and dog images (2 classes)
    cat_dog_train_mask = [True if (a == 3 or a == 5) else False for a in train_labels]
    cat_dog_test_mask = [True if (a == 3 or a == 5) else False for a in test_labels]

    train_features = np.array([a for (idx, a) in enumerate(train_features) if cat_dog_train_mask[idx]])
    train_labels = np.array([a // 4 for (idx, a) in enumerate(train_labels) if cat_dog_train_mask[idx]])

    test_features = np.array([a for (idx, a) in enumerate(test_features) if cat_dog_test_mask[idx]])
    test_labels = np.array([a // 4 for (idx, a) in enumerate(test_labels) if cat_dog_test_mask[idx]])

    return (train_features, train_labels), (test_features, test_labels)


if __name__ == '__main__':
    main()
