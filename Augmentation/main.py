import random
import time
import Augmentation
import Interface
import os


def gui_work():
    """
    Запускает графический интерфейс для настройки параметров и выполняет обработку изображений на основе введённых
    данных.

    :return: Успешность выполнения.
    """
    try:
        Interface.create_gui()
        params = Interface.on_button_clicked()
        Img_Processor = Augmentation.Images()
        Img_Processor.open_folder(params['path_in'])
        Img_Processor.augmentation_random_parallel(params)
        dir = os.path.dirname((params['path_out']))
        if not os.path.exists(dir):
            os.makedirs(dir)
        Img_Processor.save_to(params['path_out'])
        Img_Processor.clear()
        return True
    except KeyboardInterrupt:
        print('Прервано пользователем')
        return False

    except Exception as e:
        print('Ошибка в ходе выполнения:', e, "\n\nПопробуйте ещё раз\n\n")
        gui_work()



#ToDo
"""
- Логирование
- Дополнительные тесты
"""

if __name__ == "__main__":
    gui_work()

    # Img_Processor = Augmentation.Images()
    # Interface.create_gui()
    # params = Interface.on_button_clicked()
    #
    # # params = {
    # #     'multiplicator': 2,
    # #     'blur': {
    # #         'enable': True,
    # #         'power_x': (2, 7),
    # #         'power_y': (2, 7)
    # #     },
    # #     'brightness': {
    # #         'enable': True,
    # #         'range': (50, 200)  # %
    # #     },
    # #     'flip': {
    # #         'enable': True,
    # #         'flip_code': (-1, 0, 1)  # horisontal, no, verical
    # #     },
    # #     'saturation': {
    # #         'enable': True,
    # #         'range': (50, 200)  # %
    # #     },
    # #     'noise': {
    # #         'enable': True,
    # #         'mean_range': (4, 16),
    # #         'variance_range': (4, 32)
    # #     },
    # #     'contrast': {
    # #         'enable': True,
    # #         'range': (50, 200)  # %
    # #     },
    # #     'crop': {
    # #         'enable': True,
    # #         'random': True,
    # #         'left': (4, 128),
    # #         'top': (4, 128),
    # #         'window_width': (64, 256),
    # #         'window_height': (64, 256)
    # #     },
    # #     'resize': {
    # #         'enable': True,
    # #         'width_range': (512, 1024),
    # #         'height_range': (512, 1024)
    # #     },
    # #     'rotate': {
    # #         'enable': True,
    # #         'angle_range': (-15, 15)
    # #     },
    # #     'text': {
    # #         'enable': True,
    # #         'font_scale_range': (3, 21),
    # #         'text': 'From the point of view of banal erudition, each arbitrarily selected predicative absorbing object \n\
    # #         of rational mystical induction can be discretely determined with the application of a situational paradigm\n\
    # #          of a communicative-functional type in the presence of a detector-archaic distributive image in the Gilbert \n\
    # #           convergence nom space,however, with a parallel collaboration analysis of spectrographic sets isomorphically\n\
    # #            relative to multiband hyperbolic paraboloids interpreting the anthropocentric Neo-Lagrange polynomial,\n\
    # #             positional significatism of the gentile theory of psychoanalysis arises,as a result of which the\n\
    # #              followingmust be taken into account since not only the esoteric, but also the existential apperceptional\n\
    # #              anthropologist, antecedently passivized by a highly material substance, has a prismatic idiosynchration,\n\
    # #               but since the valency factor is negative, then, accordingly,antagonistic discreditism degrades in the\n\
    # #                exhibition direction, since, being in a prepubertal state, almost every subject, melancholy aware of\n\
    # #                 embryonic claustrophobia, can extrapolate any process of integration and differentiation in both\n\
    # #                  directions, it follows that as a result of syn chronicity,limited by the minimum permissible\n\
    # #                   interpolation of the image, all methods of the convergent concept require almost traditional\n\
    # #                    transformations of neocolonialism.'
    # #     }
    # # }
    #
    #
    #
    # Img_Processor.clear()
    # Img_Processor.open_folder('input/')
    #
    # t1 = time.time()
    # Img_Processor.augmentation_random_parallel(params)
    # Img_Processor.save_to('path_to_output_directory/out')
