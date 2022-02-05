import numpy as np
from numpy.lib.stride_tricks import as_strided

A = np.array([])

def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):

    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0]) // stride + 1,
                    (A.shape[1] - kernel_size[1]) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size[0], kernel_size[1])
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))

A_amps = A[:,:,0] #matrix with only amplitudes
A_grds = A[:,:,1] #matrix with only gradients

reduce_pct = 0.2 #percentage of reduction of the matrix

A_amps_reduced = pool2d(A_amps, kernel_size=(3,3), stride=1, padding=0, pool_mode='avg') #reduced matrix with amplitudes
A_grds_reduced = pool2d(A_grds, kernel_size=(3,3), stride=1, padding=0, pool_mode='avg') #reduced matrix with gradients

#new reduced matrix
A_reduced = np.zeros((A_amps.shape[0], A_amps.shape[1], 2))
A_reduced[:,:,0] = A_amps_reduced
A_reduced[:,:,1] = A_grds_reduced

output_shape = A.shape[0]*(1-reduce_pct), A.shape[1]*(1-reduce_pct)
kernel_size = (2,2)
stride = (2,2)







