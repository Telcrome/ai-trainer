from trainer.ml import Dataset
# from ml.models.unet_mod import compile_and_train

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from trainer.ml.models.deeplab import get_model, predict as dl_predict, compile_and_train as dl_compile_and_train

print("INITIALIZED ALL AI STUFF")

model = get_model(tf.keras.utils.get_file('bone_segmentation_binary_weights.h5',
                                          'https://rwth-aachen.sciebo.de/s/s7wR70RG85PMwM6/download'))


def compile_and_train(ds: Dataset, structure_name='bone'):
    return dl_compile_and_train(ds, model, structure_name=structure_name)


def predict(im: np.ndarray, ds: Dataset, structure_name: str):
    return dl_predict(model, im, ds, structure_name)


if __name__ == '__main__':
    import imageio

    im = imageio.imread("../sample_data/example_image.png")
    result = predict(im)
    sns.heatmap(result)
    plt.show()
    plt.imshow(im, cmap=plt.cm.gray)
    plt.show()
