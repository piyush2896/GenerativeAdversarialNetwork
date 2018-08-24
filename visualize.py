from matplotlib import pyplot as plt
import numpy as np
from utils import Config

def plot_generated_images(epoch, generator,
                          examples=100, dim=(10, 10),
                          figsize=(10, 10), show=False):
    noise = np.random.normal(0, 1, size=[examples, Config['gen_in_dim']])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('gan_generated_image/epoch_%d.png' % epoch)
