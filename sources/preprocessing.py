import torch
import numpy as np

def phi(seq,
        stack_num=4,
        stack_img_h=84,
        stack_img_w=84):
        
    """
    Assume the input image size is (210, 160, 3)
    Use avg_pool
    """
    img_shape = np.shape(seq[-1])
    img_stack = np.zeros((stack_num, stack_img_h, stack_img_w))
    filter_h = int(img_shape[0] / stack_img_h)
    filter_w = int(img_shape[1] / stack_img_w)
    stride_h = filter_h
    stride_w = filter_w
    
    current_gray_image = np.zeros((img_shape[0], img_shape[1]))
    current_processed_img = np.zeros((stack_img_h, stack_img_w))
    
    if 2 * stack_num - 1 > len(seq):
        stack_num_return = int((len(seq) + 1)/2)
    else:
        stack_num_return = stack_num
        
    for idx in range(1, 1+2*stack_num_return, 2):
        current_frame = seq[0-idx]
        prev_frame = seq[0-(idx+2)]
        
        for i in range(img_shape[0]):
            for j in range(img_shape[1]):
                current_frame[i][j][0] = max(current_frame[i][j][0], prev_frame[i][j][0])
                current_frame[i][j][1] = max(current_frame[i][j][1], prev_frame[i][j][1])
                current_frame[i][j][2] = max(current_frame[i][j][2], prev_frame[i][j][2])
                
                # RGB -> Y: Y = 0.2126*R + 0.7152*G + 0.0722*B
                
                current_gray_image[i][j] = 0.2126 * current_frame[i][j][0] + 0.7152 * current_frame[i][j][1] + 0.0722 * current_frame[i][j][2]
        
        for ii in range(stack_img_h):
            for jj in range(stack_img_w):
                pixel_block = np.zeros((filter_h, filter_w))
                
                if (ii * stride_h + filter_h <= img_shape[0]) and (jj * stride_w + filter_w <= img_shape[1]):
                    pixel_block = current_gray_image[ii * stride_h : ii * stride_h + filter_h, jj * stride_w : jj * stride_w + filter_w]
                elif (ii * stride_h + filter_h <= img_shape[0]) and (jj * stride_w + filter_w > img_shape[1]):
                    pixel_block[0:filter_h, 0:img_shape[1]-jj*stride_w] = current_gray_image[ii * stride_h : ii * stride_h + filter_h, jj * stride_w : img_shape[1]]
                elif (ii * stride_h + filter_h > img_shape[0]) and (jj * stride_w + filter_w <= img_shape[1]):
                    pixel_block[0:img_shape[0] - ii*stride_h, 0:filter_w] = current_gray_image[ii * stride_h : img_shape[0], jj * stride_w : jj * stride_w + filter_w]
                else:
                    pixel_block[0:img_shape[0] - ii * stride_h, 0:img_shape[1] - jj * stride_w] = current_gray_image[ii * stride_h : img_shape[0], jj * stride_w : img_shape[1]]
                    
                current_processed_img[ii][jj] = pixel_block.sum()/pixel_block.size
                
        img_stack[np.int64((idx-1)/2)] = current_processed_img
        print(img_stack)
    
    return torch.from_numpy(img_stack)
                
        
        

    


