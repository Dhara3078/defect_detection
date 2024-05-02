import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle


class EngineFaultDetection:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        return df

    def split_data(self, df,test_size=0.2):
        X_train, X_test, y_train, y_test= train_test_split(df.drop(columns=['Fault']), df['Fault'], test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def scale_data(self,X_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    # Add methods for other models (Random Forest, SVC, KNN, ANN) similarly

    def build_neural_network(self,shape):
        model = Sequential([
            Dense(64, input_shape=(shape,), activation='relu'),
            Dense(64, activation='relu'),
            Dense(4, activation='softmax')
        ])
        return model


    def train_neural_network(self):
        df=self.load_data()
        X_train, X_test, y_train, y_test=self.split_data(df)
        X_train_scaled, X_test_scaled=self.scale_data(X_train,X_test)
        model=self.build_neural_network(X_train.shape[1])
        # Train other models similarly
        callback = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        history =model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, callbacks=[callback])
        self.plot_history(history)
        file_name=os.path.join("engine","model_inference","engine_model.pkl")
        self.save_model(model,file_name)
        
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

    def save_model(self,model,filename):
        """ Saves the CNN Model in pickle file """
        pickle.dump(model, open(filename, 'wb'))


if __name__ == "__main__":
    file_path= os.path.join("engine","dataset","EngineFaultDB_Final.csv")
    efd = EngineFaultDetection(file_path)
    efd.train_neural_network()
