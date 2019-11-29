import torch
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def phi(seq,
        stack_num=4,
        stack_img_h=84,
        stack_img_w=84):
        
    """
    Assume the input image size is (210, 160, 3)
    Use skimage package to process image: rgb2gray and resize
    """
    img_shape = np.shape(seq[-1])
    img_stack = np.zeros((stack_num, stack_img_h, stack_img_w))

    current_processed_img = np.zeros((stack_img_h, stack_img_w))
    
    if 2 * stack_num - 1 > len(seq):
        stack_num_return = int((len(seq) + 1)/2)
        print("Warning: The image frames in the seqence is less than the required stack number!!!")
    else:
        stack_num_return = stack_num
        
    for idx in range(1, 1+2*stack_num_return, 2):
        current_frame = seq[0-idx]
        
        
        if len(seq) > 1:
            prev_frame = seq[0-(idx+2)]
        else:
            prev_frame = np.zeros(img_shape)
       
        current_frame = np.maximum(current_frame, prev_frame)
        current_processed_img = resize(rgb2gray(current_frame), (stack_img_h, stack_img_w), anti_aliasing=True)
        img_stack[np.int64((idx-1)/2)] = current_processed_img
        
    
    return (img_stack-0.5)/0.5
                
        
        

    


