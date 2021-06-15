import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle as pkl
import pathlib
import pandas as pd

data_dir = '/Users/burakcivitcioglu/Documents/ising_data_test/'
batch_size = 32
img_height = 100
img_width = 100


test_path = '/Users/burakcivitcioglu/Documents/ising_data_test/SQ_L_100_J_1.00_h_0.00_T_9.00/SQ_L_100_J_1.00_h_0.00_T_9.00_s_1154_n_0.txt'
test_txt = tf.io.read_file(test_path)
print(type(test_txt))
print(test_txt.numpy())

test_txt_np = test_txt.numpy()

print(float(test_txt.numpy().split()[2]))

test_img = tf.io.read_file('/Users/burakcivitcioglu/Documents/ising_data_test/SQ_L_100_J_1.00_h_0.00_T_9.00/SQ_L_100_J_1.00_h_0.00_T_9.00_s_1154_n_0.png')
image = tf.image.decode_png(test_img, channels=3)
type(image)
plt.imshow(image)

image_path_list = sorted(glob(data_dir + '*/*.png'))
img_data = tf.data.Dataset.list_files(image_path_list)
txt_path_list = sorted(glob(data_dir + '*/*_n*[0].txt'))
txt_data = tf.data.Dataset.list_files(txt_path_list)

for elem in img_data:
    print(elem)
    