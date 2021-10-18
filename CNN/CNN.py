import numpy as np
import matplotlib.pyplot as plt

#이미지 가장자리에 0으로 된 테두리(패딩)
def zero_pad(X, pad):
    '''
    X: python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad: integer, amount of padding around each image on vertical and horizontal dimensions
    '''
    
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)),
                   mode='constant', constant_values=(0,0))

    return X_pad

#단일 컨볼루션 연산 구현
def conv_single_step(a_slice_prev, W, b):
    '''
    a_slice_prev: slice of input data of shape (f, f, n_C_prev)
    W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b: Bias parameters contained in a window - matrix of shape (1, 1, 1)
    '''
    
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = Z + float(b)

    return Z

#컨볼루션 연산 정방향 연
def conv_forward(A_prev, W, b, hparameters):
    '''
    A_prev: output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W:Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b: Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters: python dictionary containing "stride" and "pad"
    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape[0], A_prev.shape[1], A_prev.shape[2], A_prev.shape[3]
    (f, f, n_XC_prev, n_C) = W.shape[0], W.shape[1], W.shape[2], W.shape[3]
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    Z = np.zeros([m, n_H, n_W, n_C])
    
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range(n_C):
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]
                    
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

#풀링 레이어 구현 (mode를 통해 MaxPooling, AveragePooling 변환 가능)
def pool_forward(A_prev, hparameters, mode='max'):
    '''
    A_prev: Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters: python dictionary containing "f" and "stride"
    mode: the pooling mode you would like to use, defined as a string ("max" or "average")
    '''
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):
        for h in range(n_H):
            vert_start = stride * h
            vert_end = vert_start + f
            
            for w in range(n_W):
                horiz_start = stride * w
                horiz_end = horiz_start + f
                
                for c in range(n_C):
                    a_prev_slice = A_prev[i]
                    
                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice[vert_start:vert_end, horiz_start:horiz_end, c])
                    elif mode == "average":
                        A[i, h, w, c] = np.average(a_prev_slice[vert_start:vert_end, horiz_start:horiz_end, c])

    cache = (A_prev, hparameters)
    
    return A, cache

#컨볼루션 연산 역방향 연산
def conv_backward(dZ, cache):
    '''
    dZ: gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache: cache of values needed for the conv_backward(), output of conv_forward()
    '''
    
    (A_prev, W, b, hparameters) = cache

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]   
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]  
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice_prev * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    return dA_prev, dW, db

#행렬의 최대값 추적 -> mask matrix (MaxPooling에 이용)
def create_mask_from_window(x):
    '''
    x: Array of shape(f, f)
    '''
    
    mask = (x == np.max(x))
    
    return mask

#dZ값을 균등하게 분배 -> AveragePooling에 이용
def distribute_value(dz, shape):
    '''
    dz: input scalar
    shape: the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    '''
    
    (n_H, n_W) = shape

    average = dz / (n_H * n_W)

    a = np.ones((n_H, n_W)) * average
    
    return a

#풀링 레이어에 대한 역방향 연산
def pool_backward(dA, cache, mode = "max"):
    '''
    dA: gradient of cost with respect to the output of the pooling layer, same shape as A
    cache: cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode: the pooling mode you would like to use, defined as a string ("max" or "average")
    '''
    (A_prev, hparameters) = cache

    stride = hparameters['stride']
    f = hparameters["f"]

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (m, n_H, n_W, n_C) = dA.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    
    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    if mode == 'max':
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = create_mask_from_window(a_prev_slice) 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[i, h, w, c])                 
                    elif mode == 'average':
                        da = dA[i, h, w, c]
                        shape = (f,f)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
    return dA_prev
