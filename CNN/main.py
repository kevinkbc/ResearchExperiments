import tensorflow as tf
import json
import numpy as np

from data_utils import Data
from models.char_cnn_zhang import CharCNNZhang
from models.char_cnn_kim import CharCNNKim

from keras.backend.tensorflow_backend import set_session

#limit gpu vram usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

tf.flags.DEFINE_string("model", "char_cnn_zhang", "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("config.json"))
    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data()
    training_inputs, training_labels = training_data.get_all_data()
    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data()
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load model configurations and build model
    modelName = config["model"]
    print(modelName)
    print(FLAGS.model)
    if FLAGS.model == "kim":
        modelName = "char_cnn_kim"
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config[modelName]["embedding_size"],
                           conv_layers=config[modelName]["conv_layers"],
                           fully_connected_layers=config[modelName]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config[modelName]["dropout_p"],
                           optimizer=config[modelName]["optimizer"],
                           loss=config[modelName]["loss"])
    else:
        modelName = config["model"]
        model = CharCNNZhang(input_size=config["data"]["input_size"],
                             alphabet_size=config["data"]["alphabet_size"],
                             embedding_size=config[modelName]["embedding_size"],
                             conv_layers=config[modelName]["conv_layers"],
                             fully_connected_layers=config[modelName]["fully_connected_layers"],
                             num_of_classes=config["data"]["num_of_classes"],
                             threshold=config[modelName]["threshold"],
                             dropout_p=config[modelName]["dropout_p"],
                             optimizer=config[modelName]["optimizer"],
                             loss=config[modelName]["loss"])
    # Train model
    model.train(training_inputs=training_inputs,
                training_labels=training_labels,
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])

from sklearn.metrics import classification_report, accuracy_score

#metrics
predictions = model.model.predict(validation_inputs)
classPredictions = np.argmax(predictions,axis=-1)
classValidationLabels = np.argmax(validation_labels,axis=-1)
print("predictions\n")
print(classPredictions)
print("labels\n")
print(classValidationLabels)

print(classification_report(classValidationLabels, classPredictions))

print("Accuracy\n")
print(accuracy_score(classValidationLabels, classPredictions))
