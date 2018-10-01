
# ---------------------
# Import Libraries
# ---------------
import numpy as np
from DIP_Framework import *
from src.utils.inpainting_utils import *
from src.utils.utils import *
import matplotlib.pyplot as plt


# ---------------------
# Load Image
# ---------------
imsize = -1
img_type = 'grayscale'
img_name = 'mnist_example.png'
img_dir = '../data/common_photos/'
img_path = img_dir + img_name
_, img_np = get_image(img_path, imsize, img_type)

# ---------------------
# Create DIP network / object
# ---------------
in_shape = 128
dip = MNIST_DIP_net(img_np, in_shape, img_name)


# ---------------------
# Phase Retrieval
# ---------------

# mn_ratios = np.linspace(0.05, 0.9, 10)
# A_types = ["gaussian", "c_gaussian"]
# dip.sweep_m_comparison(mn_ratios, A_types, "phase-retrieval", 501)

# mn_ratios = np.linspace(0.05, 0.9, 5)
# dip.sweep_m_plot(mn_ratios, "gaussian", "phase-retrieval", 300, plot_flag=False, save_flag=True)

dip.reconstruct("gaussian", "phase-retrieval", 0.2, 501, True)

# #---------------------
# # Save Grayscale Image
# #-----------------
#
# x = img_np.reshape(img_np.shape[1:])
# plt.figure(figsize=(50, 50))
# plt.imshow(x, cmap='gray', interpolation='lanczos')
# full_directory = dip.plot_dir + "gaussian" + "/"
# plt.savefig(full_directory + "original.png")