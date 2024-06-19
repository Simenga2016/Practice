import Augmentation

# img = Augmentation.Image('test1.jpg')
#
# # img.add_noise(16,64)
# # img.change_saturation(2)
# # img.visualize_image()
# img.visualize_image()
# aug = [
#     [(img.random_crop_image,(320,100)),(img.add_noise,(16,64)),(img.change_saturation,(2,))],
#     [(img.add_noise,(100,64)),(img.change_contrast,(0.6,)),(img.random_crop_image,(640,200))],
#     [(img.add_noise,(4,100)),(img.change_saturation,(0.2,)),(img.random_crop_image,(640,500))],
#     # [(img.add_noise,(16,64)),(img.random_crop_image,(32,320))]
# ]
#
# out = img.multy_augment(aug)
# img.visualize_image()
# for i in out:
#     print(i, img)
#     i.visualize_image()
#     img.visualize_image()

Img_Processor = Augmentation.Images()

Img_Processor.open_folder('.')
aug = [('resize_image', (1024, 1024)), ('random_crop_image', (328, 256)), ('resize_image', (256, 256)),
       ('change_saturation', (1.8,)), ('add_noise', (16, 64)), ('change_brightness', (0.59,))]
# for img in (Img_Processor.augmentation_all(aug)):
#     img.visualize_image()
Img_Processor.augmentation_all(aug)
Img_Processor.save_to('path_to_output_directory/out')

print(*Img_Processor.random_params.get('brightness', {}).get('range', False))