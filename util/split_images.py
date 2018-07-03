import os
import image_slicer
import sys

image_slicer.slice('../graph.png', 4, save=False)


cameras = ['c0','c1','c2','c3']
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create train image folders
train_data = '/home/sid/rddnn/train_dir/'
train_folder = '/home/sid/rddnn/split_train'
for cam in cameras:
	if not os.path.isdir(os.path.join(train_folder, cam)):
		cam_folder = os.path.join(train_folder, cam)
		os.makedirs(cam_folder)
	for i in range(10):
		folder = os.path.join(train_folder, cam, objects[i])
		if not os.path.isdir(folder):
			os.makedirs(folder)

for obj in objects:
	working_dir = os.path.join(train_data, obj)
	imgs = os.listdir(working_dir)
	print(working_dir)
	count = 0
	for img in imgs:
	# split is now an array of quarters of the img
		split = image_slicer.slice(os.path.join(working_dir, img), 4, save=False)
		image_slicer.save_tiles(split, directory='/home/sid/rddnn/split_train/'+obj, prefix=str(img))
		print(str(count) + '/' + str(len(imgs)), end='\r')
		count += 1
'''
        
# Create test image folders
test_folder = '/home/sid/rddnn/split_val'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
	for i in range(10):
		folder = os.path.join(data_dir, test_folder, objects[i])
		os.makedirs(folder)
'''
