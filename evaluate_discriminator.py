import numpy as np
from utils import *
from nn import get_optimizer
import os
from visualize import plot_generated_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def evaluate():
    print(' * Collecting Real Data')
    (x_train, y_train), (x_test, y_test) = load_data()

    print(' * Collecting Optimizer')
    adam = get_optimizer()

    print(' * Collecting Generator Model')
    generator = load_model('models/generator', adam)

    print(' * Collecting Discriminator Model')
    discriminator = load_model('models/discriminator', adam)

    print(' * Generating 100 examples')
    noise = np.random.normal(0, 1, size=[100, Config['gen_in_dim']])
    generated_images = generator.predict(noise)

    print(' * Concating 100 real and 100 fake images')
    X = np.concatenate([x_train[np.random.randint(x_train.shape[0], size=100)], generated_images])
    y = np.zeros(X.shape[0])
    y[:100] = 1

    print(' * Evaluating Discriminator')
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    res = discriminator.evaluate(X[indices], y[indices])
    print('Loss: ' + str(res[0]))
    print('Accuracy: ' + str(res[1]))

if __name__ == '__main__':
    evaluate()
