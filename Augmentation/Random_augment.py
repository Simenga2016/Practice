import random

def update_parameters(random_params, params):
    """
    Обновляет random_params согласно переданным параметрам.

    :param random_params: Текущие случайные параметры.
    :param params: Новые параметры для обновления.
    """
    if isinstance(params, dict):
        for key, value in params.items():
            if key in random_params:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key in random_params[key]:
                            random_params[key][sub_key] = sub_value
                else:
                    random_params[key] = value

def process_blur(img, params):
    if params.get('enable', False):
        power_x = random.randint(*params.get('power_x', (11, 11)))
        power_y = random.randint(*params.get('power_y', (11, 11)))
        power_y += 1 if power_y % 2 == 0 else 0
        power_x += 1 if power_x % 2 == 0 else 0
        img.blur_image(power_x, power_y)

def process_brightness(img, params):
    if params.get('enable', False):
        brightness = random.randint(*params.get('range', (100, 100))) / 100
        img.change_brightness(brightness)

def process_flip(img, params):
    if params.get('enable', False):
        flip = random.choice(params.get('flip_code', [0]))
        if flip == -1:
            img.flip_horizontal()
        elif flip == 1:
            img.flip_vertical()

def process_saturation(img, params):
    if params.get('enable', False):
        saturation = random.randint(*params.get('range', (50, 50))) / 100
        img.change_saturation(saturation)

def process_noise(img, params):
    if params.get('enable', False):
        var = random.randint(*params.get('variance_range', (4, 4)))
        mean = random.randint(*params.get('mean_range', (4, 4)))
        img.add_noise(var, mean)

def process_contrast(img, params):
    if params.get('enable', False):
        contrast = random.randint(*params.get('range', (100, 100))) / 100
        img.change_contrast(contrast)

def process_crop(img, params):
    if params.get('enable', False):
        x = random.randint(*params.get('left', (0, 0)))
        y = random.randint(*params.get('top', (0, 0)))
        wid = random.randint(*params.get('window_width', (0, 0)))
        hei = random.randint(*params.get('window_height', (0, 0)))
        if params.get('random', False):
            img.random_crop_image(wid, hei, x, y)
        else:
            img.crop_image(x, y, wid, hei)

def process_resize(img, params):
    if params.get('enable', False):
        wid = random.randint(*params.get('width_range', (512, 512)))
        hei = random.randint(*params.get('height_range', (512, 512)))
        img.resize_image(wid, hei)

def process_rotate(img, params):
    if params.get('enable', False):
        angle = random.randint(*params.get('angle_range', (0, 0)))
        img.rotate_image(angle)

def process_text(img, params):
    if params.get('enable', False):
        text = params.get('text', ' ')
        for txt in text.split('\n'):
            position = (
                random.randint(*params.get('position_x_range', (0, 0))),
                random.randint(*params.get('position_y_range', (0, 0)))
            )
            scale = random.randint(*params.get('font_scale_range', (10, 10))) / 10
            color_tmp = params.get('color_range', ((255, 255), (255, 255), (255, 255)))
            color = (
                random.randint(*color_tmp[0]),
                random.randint(*color_tmp[1]),
                random.randint(*color_tmp[2])
            )
            thickness = random.randint(*params.get('thickness_range', (1, 1)))
            blur = params.get('enable_blur', False)
            power_tmp = params.get('blur_range', ((3, 37), (3, 37)))
            power = (
                random.randint(*power_tmp[0]),
                random.randint(*power_tmp[1])
            )
            angle = random.randint(*params.get('angle_range', (-15, 15)))

            img.add_text_to_image(txt, position, scale, color, thickness, blur, power, angle)
