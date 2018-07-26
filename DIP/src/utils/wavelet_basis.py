"""Create and save the wavelet basis matrix"""
# pylint: disable = C0103

import pywt
import numpy as np


def generate_basis(x=np.zeros((10, 10))):
    """generate the basis"""
    coefs = pywt.wavedec2(x, 'db1')
    n_levels = len(coefs)
    basis = []
    for i in range(n_levels):
        coefs[i] = list(coefs[i])
        n_filters = len(coefs[i])
        for j in range(n_filters):
            for m in range(coefs[i][j].shape[0]):
                try:
                    for n in range(coefs[i][j].shape[1]):
                        coefs[i][j][m][n] = 1
                        temp_basis = pywt.waverec2(coefs, 'db1')
                        basis.append(temp_basis)
                        coefs[i][j][m][n] = 0
                except IndexError:
                    coefs[i][j][m] = 1
                    temp_basis = pywt.waverec2(coefs, 'db1')
                    basis.append(temp_basis)
                    coefs[i][j][m] = 0

    basis = np.array(basis)
    return basis


def main():
    x = np.zeros((28,28))
    basis = generate_basis(x)
    dir = "./../../data/misc/"
    np.save(dir+'wavelet_basis.npy', basis)


if __name__ == '__main__':
    main()
