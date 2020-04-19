import tensorflow as tf

def convert_h5_to_tflite(path_to_model, path_to_new_model):
    converter = tf.lite.TFLiteConverter.from_keras_model(path_to_model)
    tflite_model = converter.convert()
    print("Converting model...")
    open(path_to_new_model, "wb").write(tflite_model)
    print("Finished.")

# binary_model
PATH_BIN = "C:/Users/Natalia/PycharmProjects/ocr/models/cnn_basic/binary_model/binary_model_final.h5"
PATH_BIN_TFLITE = "C:/Users/Natalia/PycharmProjects/ocr/models/cnn_basic/binary_model/new_binary_model.tflite"

model = tf.keras.models.load_model(PATH_BIN)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print("Converting model...")
file = open(PATH_BIN_TFLITE, "wb")
file.write(tflite_model)
print("Finished.")







# convert_h5_to_tflite(PATH_BIN, PATH_BIN_TFLITE)
# new_model = keras.models.load_model(PATH_BIN)

# new_model = tf.keras.models.load_model(PATH_BIN)