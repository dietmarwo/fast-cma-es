
# Material flow planning for a simple a plant consisting of 2 machines.
# See https://ecosystem.siemens.com/softwareforinnovators/plant-simulation-challenge-task-3/overview

from fcmaes import retry, decpp, bitecpp
from fcmaes.optimizer import wrapper, logger, Bite_cpp
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp 
import numpy as np
from collections import deque

# The buffer between the two machines has a capacity of 8 parts. 
# It can be filled and parts can be removed without any loss of time.
buffer_capacity = 8 

# For setting the machine up from product A to product B and from product B to product A, 
# a setup time of 10 seconds is required. Setup is carried out automatically when switching between the part types.
setup_time = 10

class Machine:
    
    def __init__(self, type, dtime):
       self.product = -1
       self.last_product = -1
       self.time = 0
       self.type = type
       self.dtime = dtime
    
    def process(self, time, product):
        #print(f'process p{product} {self}')
        time = max(time, self.time)
        if product != self.last_product:
            time += setup_time
        self.product = product  
        self.time = time + self.dtime[self.type, product]
          
    def take(self):
        #print(f'take {self}')
        self.last_product = self.product
        self.product = -1
        return self.last_product
 
    def finished(self, time):
        return time >= self.time and self.product >= 0
   
    def available(self, time):
        return time >= self.time and self.product < 0
    
    def __str__(self):
        return f'm{self.type} p{self.product} t{self.time}'
        
class Buffer:

    def __init__(self):
        self.queue = deque()
       
    def add(self, product):
        #print(f'add p{product} {self}')
        self.queue.append(product)
        
    def retrieve(self):
        #print(f'retrieve {self}')
        return self.queue.popleft()
    
    def can_retrieve(self):
        return len(self.queue) > 0
    
    def can_add(self):
        return len(self.queue) < buffer_capacity

    def __str__(self):
        return f'b{len(self.queue)}'    
           
class Plant:

    def __init__(self, dtime):
    
        self.name = "Plant"
        self.dtime = dtime
        self.dim = len(dtime[0])
        self.bounds = Bounds([1]*self.dim, [50]*self.dim) 
        self.qd_bounds = self.bounds
        self.batch_size = None
        self.reset()

    def reset(self):
        self.machine1 = Machine(0, self.dtime)
        self.machine2 = Machine(1, self.dtime)
        self.buffer = Buffer()        
        self.count_finished = 0
        self.time = 0
        self.product = -1
        self.to_process = 0 # switch to product 1 at the beginning
        
    def __str__(self):
        return f'{self.time} {self.machine1} {self.machine2} {self.buffer} {self.count_finished}'      
        
    def next_product(self):
        #return 1
        if self.to_process > 0:
            self.to_process -= 1
        else:
            # switch product
            self.product = (self.product + 1) % self.dim
            self.to_process = self.batch_size[self.product]
            
        return self.product    
    
    def tick(self):
        #print(self)
        # from machine 1 to buffer
        if self.machine1.finished(self.time):
            if self.buffer.can_add():
                self.buffer.add(self.machine1.take())
        
        # from batch to machine1         
        if self.machine1.available(self.time):
            self.machine1.process(self.time, self.next_product())

        # take from machine 2 if ready    
        if self.machine2.finished(self.time):
            product = self.machine2.take()
            self.count_finished += 1
            #print(f'finished {product} {self.count_finished}')
           
        # from buffer to machine2         
        if self.machine2.available(self.time) and self.buffer.can_retrieve():
            self.machine2.process(self.time, self.buffer.retrieve())

        self.time += 1
           
    def simulate(self, all_time, batch_size):
        self.reset()
        self.batch_size = batch_size
        while self.time < all_time:
            self.tick()
        return self.count_finished

    def simulate_threadsafe(self, all_time, batch_size):
        plant = Plant(self.dtime)
        return plant.simulate(all_time, batch_size)
        
    def fitness(self, x):
        batch_size = x.astype(dtype=int)
        count = self.simulate_threadsafe(60*60*24, batch_size)
        return -count

def simulate(plant):
    print (plant.simulate(60*60*24, [10, 32])) # 24 hours
    
def optimize_de(plant): 
    fit = wrapper(plant.fitness)
    ret = decpp.minimize(fit, 2, plant.bounds, max_evaluations=30000, 
                         workers=mp.cpu_count(), ints=[True,True])

def optimize_bite(plant): 
    fit = wrapper(plant.fitness)
    store = retry.Store(fit, plant.bounds, logger=logger()) 
    retry.retry(store, Bite_cpp(500).minimize, num_retries=64)  
    plot3d(store, "plant")  
   
def plot3d(store, name, xlabel='batch1', ylabel='batch2', zlabel='number'):
    import matplotlib.pyplot as plt
    xs = store.get_xs()
    ys = store.get_ys()  
    x = xs[:, 0]; y = xs[:, 1]; z = ys
    fig = plt.figure()
    ax = fig.add_subplot()     
    img = ax.scatter(x, y, s=4, c=z, cmap='rainbow')
    cbar = fig.colorbar(img)
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)
    cbar.set_label(zlabel)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    fig.savefig(name, dpi=300)

# On machine 1, product A has a processing time of 10 seconds and product B has a processing time of 20 seconds.
# On machine 2, product A has a production time of 25 seconds and product B has a processing time of 15 seconds.       

dtime_orig = np.array(
    [[10, 20],
     [25, 15]]   
    )

dtime_2 = np.array(
    [[10, 20, 13, 23, 12],
     [25, 15, 23, 17, 19]]   
    )
    
def main():
    optimize_bite(Plant(dtime_orig))
    #optimize_de(Plant(dtime_orig))
    #simulate(Plant(dtime_orig))
    
if __name__ == '__main__':
    main()