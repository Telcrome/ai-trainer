from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from trainer.ml.generator import resize, batcherize, normalize_im
from trainer.ml.models import pix2pix

from trainer.ml import Dataset


# Benchmark the generator
def benchmark(g):
    ts = []
    for t in tqdm(g):
        ts.clear()
        ts.append(t)


# Visualize
def viz():
    vid, gt, frame = next(g)
    print(vid.shape)
    print(gt.shape)
    print(frame)
    im = vid[frame, :, :, :]

    plt.subplot(2, 1, 1)
    plt.imshow(im)
    plt.subplot(2, 1, 2)
    plt.imshow(gt)
    plt.show()


base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]

OUTPUT_CHANNELS = 2
# This is the last layer of the model
last = tf.keras.layers.Conv2DTranspose(
    OUTPUT_CHANNELS, 3, strides=2,
    padding='same', activation='softmax')  # 64x64 -> 128x128

inputs = tf.keras.layers.Input(shape=[128, 128, 3])
x = inputs

# Downsampling through the model
skips = down_stack(x)
x = skips[-1]
skips = reversed(skips[:-1])

# Upsampling and establishing the skip connections
for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

x = last(x)

model = tf.keras.Model(inputs=inputs, outputs=x)

tfds.disable_progress_bar()
dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def g_convert(g):
    for vid, gt, f in g:
        if f >= 2:
            res = np.zeros((vid.shape[1], vid.shape[2], 3))
            res[:, :, 0] = vid[f - 2, :, :, 0]
            res[:, :, 1] = vid[f - 1, :, :, 0]
            res[:, :, 2] = vid[f, :, :, 0]
            res_gt = np.zeros((gt.shape[0], gt.shape[1], 1), dtype=np.float32)
            res_gt[:, :, 0] = gt.astype(np.float32)
            yield normalize_im(res).astype(np.float32), res_gt


def g_appendaxis(g):
    for vid, gt in g:
        yield vid, gt[..., np.newaxis]


def get_generator_shape(g_train):
    a, b = next(g_train)
    print(a.shape)
    print(b.shape)
    print(a.dtype)
    print(b.dtype)
    return a, b


def compile_and_train(ds: Dataset, structure_name='bone'):
    g = ds.random_struct_generator(struct_name=structure_name)
    g_extracted = g_convert(g)
    g_resized = resize(g_extracted, (128, 128))
    g_train = batcherize(g_resized, batchsize=1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    EPOCHS = 5
    STEPS_PER_EPOCH = 64
    model_history = model.fit_generator(
        g_train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        # verbose=2
    )
    return model_history
# dataset_path, _ = standalone_foldergrab(folder_not_file=True, title='Pick the dataset')
#     ds = Dataset.from_disk(dataset_path)
