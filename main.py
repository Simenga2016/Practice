import Augmentation

img = Augmentation.Image('test1.jpg')

img.add_noise(16,64)
img.change_saturation(7)
img.visualize_image()

