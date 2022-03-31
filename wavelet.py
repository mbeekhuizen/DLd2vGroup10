import pywt
import numpy as np
 #Method should output 2 vectors of 100x19
def gen_wavelet(data):
    ncol = data.shape[1]
    nrow = data.shape[0]
    cur_col = data[:, 0].copy()
    (cA, cD) = pywt.dwt(cur_col, 'haar')
    # new_col1 = np.reshape(cA,(nrow/2,1))
    # new_col2 = np.reshape(cA, (nrow/ 2, 1))
    vector1 = np.reshape(cA, (len(cA), 1))
    vector2 = np.reshape(cD, (len(cD), 1))

    for i in range(1, ncol):
        cur_col = data[:,i].copy()
        (cA, cD) = pywt.dwt(cur_col, 'haar')
        vector1 = np.hstack((vector1, np.reshape(cA, (len(cA), 1))))
        vector2 = np.hstack((vector2, np.reshape(cD, (len(cD), 1))))
    return vector1, vector2


def gen_wavelet2(data):
    ncol = data.shape[1]
    nrow = data.shape[0]
    for i in range(ncol):
        cur_col = data[:,i].copy()
        (cA, cD) = pywt.dwt(cur_col, 'haar')
        new_col = np.reshape(np.concatenate((cA,cD), 0),(nrow,1))
        data = np.hstack((data,new_col))
    return data