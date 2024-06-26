import time
import Augmentation
import Interface
from fixed_params import params as fixed_params

if __name__ == "__main__":
    Img_Processor_GUI = Augmentation.Images()
    Interface.create_gui()
    # Тест интерфейса
    params = Interface.on_button_clicked()

    ## Фиксированный тест
    # params = fixed_params

    Img_Processor_GUI.clear()
    Img_Processor_GUI.open_folder('input/')

    # Временной тест
    t1 = time.time()
    Img_Processor_GUI.augmentation_random_parallel(params)
    Img_Processor_GUI.save_to('path_to_output_directory/out')
    print('parallel time: ', time.time() - t1, 'total:', len(Img_Processor_GUI.images))

    Img_Processor_GUI.clear()
    Img_Processor_GUI.open_folder('input/')

    t1 = time.time()
    Img_Processor_GUI.augmentation_random(params)
    Img_Processor_GUI.save_to('path_to_output_directory1/out')
    print('linear time:', time.time() - t1, 'total:', len(Img_Processor_GUI.images))