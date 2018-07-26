
# ---------------------
# Import Libraries
# ---------------
import numpy as np
from DIP_Framework import *
from src.utils.inpainting_utils import *
from src.utils.utils import *


# ---------------------
# Load Images
# ---------------
imsize = -1
img_type = 'grayscale'
img_path = '../data/common_photos/vase.png'
_, img_np = get_image(img_path, imsize, img_type)


# ---------------------
# Create DIP network / object
# ---------------
in_shape = img_np.shape[1:]
dip = DIP_net(img_np, in_shape)


# ---------------------
# Phase Retrieval
# ---------------
# dip.reconstruct("gaussian", 0.5, 1300)

# dip.reconstruct("DFT", "phase-retrieval", 0.5, 1300)

# mn_ratios = np.linspace(0.2, 0.9, 10)
# dip.sweep_m_plot(mn_ratios, "gaussian", "phase-retrieval", 1300, plot_flag=False, save_flag=True)

mn_ratios = np.linspace(0.2, 0.9, 10)
A_types = ["gaussian", "c_gaussian"]
dip.sweep_m_comparison(mn_ratios, A_types, "phase-retrieval", 1300)
