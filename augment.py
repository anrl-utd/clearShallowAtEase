import Augmentor

p = Augmentor.Pipeline('/home/sid/deep-margins/cifar/train/cats')
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
p.random_distortion(probability=0.8, grid_width=6, grid_height=6, magnitude=8)

p.sample(50000)
