import numpy as np

class replay_buffer():
    
    def __init__(self,
                 buffer_size:int,
                 ):
                 
        self.buffer_size = buffer_size
        self.stored_samples = 0
        self.store_ptr = 0
        
        self.buffer = []
        
    def store(self,
              img_prev:np.array,
              action:int,
              reward:float,
              img_next:np.array
              ):
              
        if self.stored_samples < self.buffer_size:
            self.buffer.append((img_prev, action, reward, img_next))
            self.stored_samples += 1
            self.store_ptr = (self.store_ptr + 1) % self.buffer_size
        else if self.stored_samples == self.buffer_size:
            self.buffer[self.store_ptr] = (img_prev, action, reward, img_next)
            self.store_ptr = (self.store_ptr + 1) % self.buffer_size
        else:
            self.stored_samples = self.buffer_size
            
    def sample(self,
               batch_size: int) -> np.array:
               
        if batch_size >= self.stored_samples:
            return np.array(self.buffer)
        else:
            idx_list = np.random.randint(0, self.stored_samples, batch_size)
            return np.array([self.buffer[idx] for idx in idx_list])
            
            
            
              
        
        
    
        
