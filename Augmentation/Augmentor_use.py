import Augmentor

p = Augmentor.Pipeline("./input",save_format='png')
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.4)
p.random_distortion(probability=1, grid_width=8, grid_height=8, magnitude=16)
p.sample(50)
