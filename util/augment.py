import Augmentor

dir_list = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']

for folder in dir_list:
	p = Augmentor.Pipeline('/home/sid/rddnn/train_dir/' + folder)
	p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
	p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
	p.random_distortion(probability=0.8, grid_width=6, grid_height=6, magnitude=8)
	
	p.sample(5000)
