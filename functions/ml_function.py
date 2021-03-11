''' from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras '''
from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

def untar_file(file,folder_name:str):
    #save file
    name = file.filename
    f = open("raw_data/"+name, "wb")
    f.write(file.file.read())
    f.close()
    #preprocessing data
    url = "file://"+os.path.abspath("./raw_data/"+name)
    fname = file.filename.split(".")[0]
    
    dataset = keras.utils.get_file(fname=fname,
                                    origin=url,
                                    untar = True,
                                    cache_dir="./train_ds",
                                    cache_subdir="")
    os.remove(os.path.abspath("train_ds/"+name))
    return {"data_dir":"train_ds/"+folder_name}

def training(train_dir:str):
    #train_ds
    train_dir = "train_ds/"+train_dir
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        batch_size=16,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(180,180)
    )
    #val_ds
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        batch_size=16,
        validation_split=0.2,
        subset='validation',
        seed = 123,
        image_size=(180,180)
    )

    #config to save model
    save_model = keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/"+train_dir.split("/")[1],
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )

    model = create_model(
        weight_dir=train_dir.split('/')[1]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=[save_model]
    )

    val_acc = max(history.history["val_accuracy"])
    return {"val_accuracy":val_acc}

def predict(data_name,weight_dir):
    sunflower_url = "file://"+os.path.abspath("predict_ds/"+data_name)
    sunflower_path = tf.keras.utils.get_file(data_name, origin=sunflower_url)

    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(180, 180)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    model = tf.keras.models.load_model("./checkpoints/"+data_name)
    result = np.argmax(model.predict(img_array))
    result = os.listdir("train_ds/"+weight_dir)[result]
    return result

def create_model(weight_dir):
    model = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(180, 180, 3)),
        keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(os.listdir("train_ds/"+weight_dir)), activation="softmax")
    ])    
    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )    
    model.summary()
    return model

def conts_train(train_dir:str, epochs:int):
    model  = tf.keras.models.load_model("./checkpoints/"+train_dir)
    #train_ds
    train_dir = "train_ds/"+train_dir
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        batch_size=16,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(180,180)
    )
    #val_ds
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        directory=train_dir,
        batch_size=16,
        validation_split=0.2,
        subset='validation',
        seed = 123,
        image_size=(180,180)
    )

    #config to save model
    save_model = keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/"+train_dir.split("/")[1],
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[save_model]
    )

    val_acc = max(history.history["val_accuracy"])
    return {"val_accuracy":val_acc}