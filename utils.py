import numpy as np
from keras.datasets import mnist
from keras.models import model_from_json

Config = {
    'gen_in_dim': 100,
    'epochs': 500,
    'batch_size': 128
}

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float') - 127.0) / 127.0
    x_train = x_train.reshape(-1, 784)

    return (x_train, y_train), (x_test, y_test)

def save_model(model, path):
    model_json = model.to_json()
    with open(path+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(path + ".h5")

def load_model(path, opt):
    json_file = open(path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path+".h5")
    loaded_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return loaded_model
