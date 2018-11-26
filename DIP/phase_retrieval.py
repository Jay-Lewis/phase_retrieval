
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
pad_dim = 128
img_type = 'grayscale'
img_name = 'cameraman_2.png'             #'black_white.gif'
img_dir = '../data/common_photos/'
img_path = img_dir + img_name
_, img_np = get_image(img_path, imsize, img_type)
true_in_shape = np.asarray(img_np.shape[1:])
in_shape_array = np.asarray(true_in_shape) + np.asarray([int(pad_dim*2), int(pad_dim*2)])
in_shape = [int(element) for element in in_shape_array]
in_shape = tuple(in_shape)
# ---------------------
# Create DIP network / object
# ---------------
dip = DIP_net(img_np, in_shape, img_name, true_in_shape)


# ---------------------
# Phase Retrieval
# ---------------

# dip.reconstruct("c_gaussian", reconstruct_mode="phase-retrieval", mn_ratio=0.6, num_iter_DIP=300, num_iter_RED=None,
#                 save_flag=True, reconstruct_strategy='normal')

# dip.reconstruct("gaussian", reconstruct_mode="phase-retrieval", mn_ratio=0.6, num_iter_DIP=301, num_iter_RED=301,
#                 save_flag=True, reconstruct_strategy='projections')

# dip.reconstruct("gaussian", reconstruct_mode="phase-retrieval", mn_ratio=0.6, num_iter=400,
#                 save_flag=True, reconstruct_strategy='normal')

dip.reconstruct("oversampled_2d_DFT_pad", "phase-retrieval", mn_ratio=1.0, num_iter_DIP=700, num_iter_RED=500,
                reconstruct_strategy='projections', save_flag=True)

# dip.reconstruct("DFT", "phase-retrieval", None, 1300)

# losses, img_losses = dip.reconstruct("oversampled_2d_DFT", "phase-retrieval", 8.0, 1300)

# mn_ratios = np.linspace(0.12, 0.4, 6)
# dip.sweep_m_plot(mn_ratios, "gaussian", "phase-retrieval", 401, plot_flag=False, save_flag=True)

# mn_ratios = np.linspace(0.05, 0.9, 2)
# A_types = ["gaussian", "c_gaussian"]
# dip.sweep_m_comparison(mn_ratios, A_types, "phase-retrieval", 10)


#---------------------
# Save Grayscale Image
#-----------------

# x = img_np.reshape(img_np.shape[1:])
# plt.figure(figsize=(50, 50))
# plt.imshow(x, cmap='gray', interpolation='lanczos')
# full_directory = dip.plot_dir + "gaussian" + "/"
# plt.savefig(full_directory + "original.png")