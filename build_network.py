import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

DATA_DIR = 'data'
# CATEGORIES = ['Lycopodiaceae', 'Selaginellaceae']
CATEGORIES = ['lyco_test_hundred', 'sela_test_hundred']
IMG_SIZE = 256 #pixels

# Open up those pickle files
features = pickle.load(open("features.pickle","rb"))
labels = pickle.load(open("labels.pickle","rb"))

# normalize data
features = features/255.0

# build the model? let's give this a shot lol
model = Sequential()

# Image input shape: 256 x 256 x 3

# 1. Convolution Layer: 10 filters of 5px by 5px
model.add(Conv2D(10, (5, 5), input_shape = (IMG_SIZE, IMG_SIZE, 3))) 
# Output shape: 10 x 252 x 252

# 2. Batch Normalization: Normalizes previous layer to have mean near 0 and S.D. near 1
model.add(BatchNormalization())
# Output shape: 10 x 252 x 252

# 3. Activation Layer: ReLU uses the formula of f(x)= x if x>0 and 0 if x<=0
# Apparently it's a pretty common one for CNN so we're going with the flow here
model.add(Activation("relu"))
# Output shape: 10 x 252 x 252

# 4. Pooling function: from the paper, it didn't specify function, but looking online, it seems that the default is Max so we are a-okay here
model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# Output shape: 10 x 126 x 126

#-------------Next Set of Layers--------------
# 5. Convolution Layer: 40 filters of 5px by 5px
model.add(Conv2D(40, (5, 5)))
# Output shape: 40 x 122 x 122

#6. Batch Normalization Layer
model.add(BatchNormalization())
# Output shape: 40 x 122 x 122

# Activation Layer: Same as above
model.add(Activation("relu"))
# Output shape: 40 x 122 x 122

#Pooling again will decrease "image shape" by half since stride = 2
model.add(MaxPooling2D(pool_size=(2,2), strides =2))
# Output shape: 40 x 61 x 61



# 2 hidden layers
model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 2 neurons, for 2 classes
model.add(Dense(2))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(features, labels, batch_size=32, epochs=10, validation_split=0.1)

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')

# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()