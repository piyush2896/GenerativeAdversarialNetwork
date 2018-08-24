from utils import *
from nn import get_optimizer
import os
from visualize import plot_generated_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def generate():
    print(' * Collecting Model')
    adam = get_optimizer()
    generator = load_model('models/generator', adam)
    print(' * Generating and Ploting 100 examples')
    plot_generated_images(0, generator, show=True)

if __name__ == '__main__':
    generate()