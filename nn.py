from keras.layers import Dense, Input, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5) # Suggested in DCGANs paper

def get_generator(in_shape, optimizer):
    x0 = Input(in_shape)
    x = Dense(256,
              kernel_initializer=initializers.RandomNormal(stddev=0.02))(x0) # Taken from DCGAN Paper
    x = LeakyReLU(0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(784, activation='tanh')(x)
    model = Model(x0, x)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def get_discriminator(in_shape, optimizer):
    x0 = Input(in_shape)
    x = Dense(1024,
              kernel_initializer=initializers.RandomNormal(stddev=0.02))(x0) # Taken from DCGAN Paper (It susggested all weights to be initialised in this manner)
    x = LeakyReLU(0.2)(x)   # Use of LeakyReLU in Discriminator taken from DCGAN paper.
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(x0, x)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

def get_gan_net(generator, discriminator, optimizer):
    x0 = generator.input
    discriminator.trainable = False
    x = discriminator(generator.output)
    model = Model(x0, x)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model
