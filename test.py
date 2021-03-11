from functions import ml_function as f
import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.load_model("./checkpoints/flower_photos")
model.summary()
