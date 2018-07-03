import os
import sys

cameras = ['c0','c1','c2','c3'] 
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
from_dir = '/home/sid/rddnn/split_train/'

for obj in objects:
	working_dir = os.path.join(from_dir, obj)
	
