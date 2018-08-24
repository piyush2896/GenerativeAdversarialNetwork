from nn import *
import numpy as np
from utils import *
from tqdm import tqdm
import os
from visualize import plot_generated_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(10)

def train():

    (x_train, y_train), (x_test, y_test) = load_data()

    n_batches = x_train.shape[0] // Config['batch_size'] + 1

    adam = get_optimizer()
    generator = get_generator([Config['gen_in_dim']], adam)
    discriminator = get_discriminator([x_train.shape[1]], adam)
    gan = get_gan_net(generator, discriminator, adam)

    for epoch in range(Config['epochs']):
        print('Epoch: {}/{}'.format(epoch, Config['epochs']))

        for _ in tqdm(range(n_batches)):
            noise = np.random.normal(0, 1, size=[Config['batch_size'], Config['gen_in_dim']])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=Config['batch_size'])]

            generated_images = generator.predict(noise)
            X = np.concatenate([image_batch, generated_images])

            y = np.zeros(2 * Config['batch_size'])
            y[:Config['batch_size']] = 0.9

            discriminator.trainable = True
            discriminator.train_on_batch(X, y)

            noise = np.random.normal(0, 1, size=[Config['batch_size'], Config['gen_in_dim']])
            y_gen = np.ones(Config['batch_size'])
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        if epoch % 20 == 0:
            plot_generated_images(epoch, generator)
            save_model(generator, 'models/generator')
            save_model(discriminator, 'models/discriminator')

if __name__ == '__main__':
    train()
