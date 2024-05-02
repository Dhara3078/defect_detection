import pickle
import os
import numpy as np
import tensorflow as tf

def predict_class(image_path):
        """ Loads the trained model from the pickle file and predicts the class of the input image """
        
        # Load the trained model
        model_path=os.path.join("metal_surface","model_inference","metal_surface_model.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load and preprocess the input image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(200, 200))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Perform prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        label_map={0:'Crazing', 1:'Inclusion', 2:'Patches', 3:'Pitted', 4:'Rolled', 5: 'Scratches'}
        return label_map[class_index]


        return class_index
if __name__ == "__main__":
      prediction=predict_class("/home/gneya/defect_detection/metal_surface/dataset/NEU Metal Surface Defects Data/test/Crazing/Cr_1.bmp")
      print(prediction)