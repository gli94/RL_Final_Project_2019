import numpy as np

class replay_buffer():
    
    def __init__(self,
                 buffer_size:int
                 ):
                 
        self.buffer_size = buffer_size
        self.stored_samples = 0
        self.store_ptr = 0
        
        self.buffer = []
        
    def store(self,
              img_prev:np.array,
              action:int,
              reward:float,
              img_next:np.array,
              done:bool
              ):
              
        """
        Each call of this method stores a 5-element tuple into the replay buffer: current processed image stack, current action taken, reward received after taking the action, next processed image stack and a flag indicates whether the episode terminates
        """
              
        if self.stored_samples < self.buffer_size:
            self.buffer.append((img_prev, action, reward, img_next, done))
            self.stored_samples += 1
            self.store_ptr = (self.store_ptr + 1) % self.buffer_size
        elif self.stored_samples == self.buffer_size:
            self.buffer[self.store_ptr] = (img_prev, action, reward, img_next, done)
            self.store_ptr = (self.store_ptr + 1) % self.buffer_size
        else:
            self.stored_samples = self.buffer_size
            
    def sample(self,
               batch_size: int) -> np.array:
               
        """
        This method returns a numpy array, with batch_size # of elements randomly sampled from replay buffer. Each element of the array is a tuple, consists of current processed image stack, action, reward, next processed image stack and a flag
        """
               
        if batch_size >= self.stored_samples:
            return np.array(self.buffer)
        else:
            idx_list = np.random.randint(0, self.stored_samples, batch_size)
            return np.array([self.buffer[idx] for idx in idx_list])
            
            
            
              
        
        
    
        
