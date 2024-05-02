import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle



""" Class for training the Metal Surface Defect"""
class MetalSurfaceDetection:
    def __init__(self, train_dir, test_dir, val_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir

    def _create_generators(self):
        """ Generates the Training, testing and validation Data"""

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode="categorical"
        )

        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode="categorical"
        )
        class_indices = train_generator.class_indices
        print("Class labels and their indices:", class_indices)
        return train_generator, validation_generator

    def train_cnn_model(self):
        """ Calls the _create_generators to get the data, calls _build_cnn_model for CNN model, trains the model on the data and save it to pickle file through save_model function"""
        train_generator, validation_generator = self._create_generators()

        model = self._build_cnn_model(train_generator.num_classes)

        checkpoint_path = "best_weights.keras"
        checkpoint = ModelCheckpoint(checkpoint_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max',
                                     )

        model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])

        history = model.fit(train_generator,
                            steps_per_epoch=train_generator.n // train_generator.batch_size,
                            epochs=15,
                            validation_data=validation_generator,
                            validation_steps=validation_generator.n // validation_generator.batch_size,
                            callbacks=[checkpoint]
                            )
        file_name= os.path.join("metal_surface","model_inference","metal_surface_model.pkl")
        print(file_name)
        self.save_model(model,file_name)
        self.plot_history(history)

    def _build_cnn_model(self, num_classes):
        """ Builds CNN Model"""
        model_cnn= Sequential()

        model_cnn.add(Conv2D(32, (3,3),input_shape=(200,200,3),activation="relu"))
        model_cnn.add(MaxPooling2D(pool_size=(2,2)))

        model_cnn.add(Conv2D(64, (3,3),input_shape=(200,200,3),activation="relu"))
        model_cnn.add(MaxPooling2D(pool_size=(2,2)))

        model_cnn.add(Conv2D(128, (3,3),input_shape=(200,200,3),activation="relu"))
        model_cnn.add(MaxPooling2D(pool_size=(2,2)))

        model_cnn.add(Flatten())
        model_cnn.add(Dense(256,activation="relu"))
        model_cnn.add(Dropout(0.3))

        model_cnn.add(Dense(num_classes,activation="softmax"))


        return model_cnn
    
    def save_model(self,model,filename):
        """ Saves the CNN Model in pickle file """
        pickle.dump(model, open(filename, 'wb'))


    def plot_history(self, history):
        """ Plots the graph of accuracy v/s Validation_accuracy and loss v/s Validation loss"""
        plt.plot(history.history["accuracy"])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Accuracy and Loss")
        plt.ylabel("Accuracy/Loss")
        plt.xlabel("Epoch")
        plt.legend(["Accuracy", "Validation Accuracy", "Loss", "Validation Loss"])
        plt.show()

if __name__ == "__main__":
    train_dir = os.path.join("metal_surface","dataset","NEU Metal Surface Defects Data","train")
    test_dir =  os.path.join("metal_surface","dataset","NEU Metal Surface Defects Data","test")
    val_dir =   os.path.join("metal_surface","dataset","NEU Metal Surface Defects Data","valid")

    msd = MetalSurfaceDetection(train_dir, test_dir, val_dir)
    msd.train_cnn_model()
