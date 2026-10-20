import numpy as np

def pad_image_zeros(img, pad_h, pad_w):
    """Pads 2D grayscale image with zeros on all 4 borders."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

def conv2d_4loops(img, kernel):
    """
    2D Convolution from scratch using 4 nested loops (Pure NumPy).
    """
    # Flip kernel 180 degrees
    k_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    
    H_in, W_in = img.shape
    H_k, W_k = kernel.shape
    pad_h, pad_w = H_k // 2, W_k // 2
    
    padded = pad_image_zeros(img, pad_h, pad_w)
    output = np.zeros_like(img, dtype=np.float64)
    
    for i in range(H_in):
        for j in range(W_in):
            acc = 0.0
            for m in range(H_k):
                for n in range(W_k):
                    acc += padded[i + m, j + n] * k_flipped[m, n]
            output[i, j] = acc
            
    return output

def conv2d_2loops(img, kernel):
    """
    2D Convolution from scratch using 2 nested loops (Vectorized NumPy slices).
    """
    k_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    
    H_in, W_in = img.shape
    H_k, W_k = kernel.shape
    pad_h, pad_w = H_k // 2, W_k // 2
    
    padded = pad_image_zeros(img, pad_h, pad_w)
    output = np.zeros_like(img, dtype=np.float64)
    
    for i in range(H_in):
        for j in range(W_in):
            patch = padded[i:i+H_k, j:j+W_k]
            output[i, j] = np.sum(patch * k_flipped)
            
    return output