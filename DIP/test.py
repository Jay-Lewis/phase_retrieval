
# ---------------------
# Import Libraries
# ---------------
import numpy as np
from DIP_Framework import *
from src.utils.inpainting_utils import *
from src.utils.utils import *
import matplotlib.pyplot as plt


# ---------------------
# Load Images
# ---------------
imsize = -1
img_type = 'grayscale'
img_path = '../data/common_photos/wave.jpg'
_, img_np = get_image(img_path, imsize, img_type)


# ---------------------
# Create DIP network / object
# ---------------
in_shape = img_np.shape[1:]
dip = DIP_net(img_np, in_shape, "who_cares")


# ---------------------
# Phase Retrieval Testing  (testing to see if my functions operate correctly for DFT measurements )
# ---------------

# Test if methods are the same
A_type = "oversampled_2d_DFT_pad"
reconstruct_mode = "phase-retrieval"
mnratio = 1.0

# DFT by tensordot
dip.A = dip.get_A(mnratio, A_type)

y_prime = np.tensordot(dip.A, img_np, 2)
A_t = np.transpose(np.reshape(dip.A, [dip.m, dip.n1]))
dft_true = np.tensordot(y_prime, A_t, 1)

# DFT by torch
dip.measurement = dip.get_measurement(A_type, reconstruct_mode)
dip.measurement_var, dip.A_var, dip.x_var, dip.real_A_var, dip.img_A_var = dip.get_torch_vars(A_type)

real_A_var_sliced = dip.real_A_var.view(dip.m, dip.n1)
img_A_var_sliced = dip.img_A_var.view(dip.m, dip.n1)
img = np_to_torch(img_np, False).type(dip.dtype)

r_r_part = tensordot_pytorch(tensordot_pytorch(dip.real_A_var, img, axes=2), real_A_var_sliced.transpose(0, 1), axes=1)
i_i_part = tensordot_pytorch(tensordot_pytorch(dip.img_A_var, img, axes=2), img_A_var_sliced.transpose(0, 1), axes=1)
i_r_part = tensordot_pytorch(tensordot_pytorch(dip.img_A_var, img, axes=2), real_A_var_sliced.transpose(0, 1), axes=1)
r_i_part = tensordot_pytorch(tensordot_pytorch(dip.real_A_var, img, axes=2), img_A_var_sliced.transpose(0, 1), axes=1)

measurement_hat = ((r_r_part - i_i_part) ** 2 + (i_r_part + r_i_part) ** 2) ** (1 / 2)
measurement_hat = measurement_hat.view(1,dip.m,dip.m)
measurement_hat = torch_to_np(measurement_hat)

print(np.shape(img_np))

real_part = torch_to_np(np.reshape(r_r_part - i_i_part, [1,dip.m,dip.m]))
img_part = torch_to_np(np.reshape(i_r_part+r_i_part, [1,dip.m,dip.m]))
dft_hat = real_part + 1j*img_part

diff = dip.measurement-measurement_hat
diff2 = dft_true - dft_hat
diff3 = np.absolute(dft_true) - dip.measurement


total = np.sum(np.absolute(diff))
print(total)
print(total/np.product(np.shape(diff)))
print(np.sum(np.absolute(dip.measurement))/np.product(np.shape(dip.measurement)))

print(dip.measurement)
print('--------------------------')
print(diff)
print('--------------------------')
print(np.min(np.abs(dip.measurement)))
print(np.max(np.abs(diff)))


newshape = img_np.shape[1:]
img_np = img_np.reshape(newshape)
plt.imshow(np.absolute(diff), vmin=0, vmax = 255, cmap='gray')
plt.show()



# # ---------------------
# # Phase Retrieval Testing (testing Tensordot of pytorch vs. np.tensordot)
# # ---------------
#
# A_type = "gaussian"
# reconstruct_mode = "phase-retrieval"
# mnratio = 1.0
# dip.A = dip.get_A(mnratio, A_type)
# dip.measurement = dip.get_measurement(A_type, reconstruct_mode)
# dip.measurement_var, dip.A_var, dip.x_var, dip.real_A_var, dip.img_A_var = dip.get_torch_vars(A_type)
# img = np_to_torch(img_np, False).type(dip.dtype)
#
# # Numpy
# y_true = np.tensordot(dip.A, img_np, 2)
#
# # Torch
# y_hat = tensordot_pytorch(dip.A_var, img, axes=2)
#
# diff = y_true-y_hat
#
# plt.imshow(diff, vmin=0, vmax = 255, cmap='gray')
# plt.show()



# # ---------------------
# # Phase Retrieval Testing  (Testing DFT matrix by fft vs. creating matrix and then using Tensordot)
# # ---------------
# #
# # Test if DFT Matrix is correct
# A_type = "oversampled_2d_DFT"
# reconstruct_mode = "phase-retrieval"
# mnratio = 1.0
#
# # DFT by matrix
# A = dip.get_A(mnratio, A_type)
#
# y_prime = np.tensordot(A, img_np, 2)
# A_t = np.transpose(np.reshape(A, [dip.m, dip.n1]))
# # dft_hat = np.tensordot(y_prime, A_t, 1)
# # fft_hat = dip.get_measurement(A_type, reconstruct_mode)
#
# # DFT by fft
# newshape = img_np.shape[1:]
# img_np = img_np.reshape(newshape)
# true_fft = np.fft.fft2(img_np)
#
# diff = true_fft - dft_hat
#
# plt.figure()
# plt.imshow(img_np, cmap='gray')
# plt.figure()
# plt.imshow(np.angle(dft_hat), cmap='gray')
# plt.figure()
# plt.imshow(np.angle(true_fft), cmap='gray')
# plt.figure()
# plt.imshow(np.abs(diff), vmin=0, vmax = 255, cmap='gray')
# plt.show()
