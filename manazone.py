import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


classes=['Akshay Kumar', 'Alexandra Daddario', 'alexis', 'Alia Bhatt',
'Amitabh Bachchan', 'Andy Samberg', 'Anushka Sharma', 'Billie Eilish',
'Brad Pitt', 'Camila Cabello', 'Charlize Theron', 'chris', 'Claire Holt', 'Courtney Cox',
'Dwayne Johnson', 'Elizabeth Olsen', 'Ellen Degeneres', 'frank', 'glory', 'Henry Cavill','Ivan',
'Lisa Kudrow', 'Margot Robbie', 'Natalie Portman', 'Priyanka Chopra', 'Robert Downey Jr',
'Roger Federer', 'Tom Cruise', 'Vijay Deverakonda', 'Virat Kohli', 'Zac Efron']

# num_image_par_classes = input(" SVP   entrer le nombre d'image de chaque classe de votre jeu de donnee:")

nipc=1

labelsEncoded= np.zeros((len(classes)*nipc, len(classes)))
for i in range(len(classes)):
    labelsEncoded[i*nipc: (i+1)*nipc,i]=1



df1=pd.DataFrame(labelsEncoded, classes)
df1.to_excel('etiquettes_oneHot.xlsx',index=False)


encoded_labels=labelsEncoded
images=classes
data_dir = "C:/Users/LAPTOP/PycharmProjects/pythonProject5/webcamCaptures/Faces"
groupes= os.listdir(data_dir)
X_train, X_val, y_train, y_val = train_test_split(images, encoded_labels, test_size=0.3, random_state=42)
print(len(X_val),len(y_val), len(X_train), len(y_train))

#Generation des images
datagen = ImageDataGenerator(
    rescale=1.0/255.,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

print(datagen)


chem="C:/Users/LAPTOP/PycharmProjects/pythonProject5/webcamCaptures/Faces/ztrain"
train_gen=ImageDataGenerator()
train_generator =train_gen.flow_from_directory( batch_size=32,
                                                          directory=chem, shuffle=True,
                                                          target_size=(224, 224),
                                                          class_mode='categorical',
                                                          seed=42)


print(train_generator)
base_model = MobileNetV2(weights="imagenet", include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(31, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



batch_size = 32
epochs = 10
model.fit(train_generator, epochs=epochs, validation_data=(X_val, y_val))



# loss, accuracy = model.evaluate(X_val, y_val)
# print(f"Précision sur l'ensemble de validation : {accuracy:.2f}")
#
#
# newImg="C:/Users/LAPTOP/PycharmProjects/pythonProject5/newIm"
# new_images = ['newImg']
# new_images = np.array([np.array(Image.open(img_path).convert("RGB").resize((224, 224))) for img_path in new_images])
#
# # Prédiction des classes pour les nouvelles images
# predictions = model.predict(new_images)
# predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
# print(predicted_classes)
#
# # Sauvegarde des résultats dans un fichier Excel
# results_df = pd.DataFrame({'Nom de l\'image': new_images, 'Classe prédite': predicted_classes})
# results_df.to_excel('visages_similaires.xlsx', index=False)
