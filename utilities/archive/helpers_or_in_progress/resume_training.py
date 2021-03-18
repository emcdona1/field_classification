import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

model = tf.keras.models.load_model("saved_models/CNN_8.13.19_13.47.model") #, custom_objects={'custom_activation':Activation(custom_activation)})

features = pickle.load(open("features.pickle","rb"))
labels = pickle.load(open("labels.pickle","rb"))

history = model.fit(features, labels, batch_size=32, initial_epoch = 20, epochs = 30, validation_split=0.22) #.22 of the .9 is .2 of the total

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

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()