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
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import keras 
from tqdm import tqdm
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')



""" Class for training the Metal Surface Defect"""
class TyreDefectDetection:
    def __init__(self, train_dir, val_dir):
        self.train_dir = train_dir
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
            class_mode="binary"
        )

        validation_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(200, 200),
            batch_size=10,
            class_mode="binary"
        )
        class_indices = train_generator.class_indices
        print("Class labels and their indices:", class_indices)
        return train_generator, validation_generator

    def train_mobilenetv3_model(self):
        """ Calls the _create_generators to get the data, calls _build_cnn_model for CNN model, trains the model on the data and save it to pickle file through save_model function"""
        train_generator, validation_generator = self._create_generators()

        model = self._build_mobilenetv3_model(train_generator.num_classes)

        checkpoint =ModelCheckpoint("model_mn.keras", save_best_only=True)
        early_stopping =EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy",)

        model.compile(optimizer ='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        history=model.fit(train_generator,epochs=5,
                       validation_data=validation_generator,
                       callbacks=[checkpoint,early_stopping],
                       validation_steps=validation_generator.n // validation_generator.batch_size,)

        file_name= os.path.join("metal_surface","model_inference","metal_surface_model.pkl")
        print(file_name)
        self.save_model(model,file_name)
        self.plot_history(history)

    def _build_mobilenetv3_model(self, num_classes):
        """ Builds CNN Model"""
        base_model = tf.keras.applications.MobileNetV3Small(input_shape=(200,200,3),include_top=False,weights='imagenet')
        base_model.trainable = False

        inputs = tf.keras.layers.Input(shape=(200, 200, 3), name="input_layer")
        x = base_model(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs=tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax,name="output_layer")(x)

        # Combine the inputs with the outputs into a model
        model_mn= tf.keras.Model(inputs, outputs, name="model")


        return model_mn
    
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
    train_dir = os.path.join("tyre","dataset","tyre_dataset","train")
    val_dir =   os.path.join("metal_surface","dataset","tyre_dataset","valid")

    msd = TyreDefectDetection(train_dir,val_dir)
    msd.train_mobilenetv3_model()
