import glob

import PySimpleGUI as sg
from hcat.lib.cell import Cell
from hcat.lib.utils import calculate_indexes
from itertools import product

from hcat.lib.cochlea import Cochlea
import skimage.io
import torchvision.utils
from PIL import Image
import io
from hcat.detect import _detect
import torch
from torch import Tensor
import numpy as np
from typing import Union, Optional, List, Dict, Tuple
import hcat.lib.utils
import torch.nn.functional as F
from torchvision.io import encode_png
from hcat.detect import _cell_nms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import torchvision.transforms.functional as ttf

from hcat.backends.detection import FasterRCNN_from_url
import os.path

MAX_WIDTH, MAX_HEIGHT = 900, 900

class gui:
    def __init__(self):
        sg.theme('DarkGrey5')
        plt.ioff()

        image_column = [
            [sg.Push(), sg.Image(filename='', key='image', size=(900, 900), enable_events=True), sg.Push()],
            [sg.Push(), sg.Text('', justification='center', key='-LOG-'), sg.Push()],
        ]

        output_column = [
            [sg.Text('File:'), sg.Push()],
            [sg.InputText('', size=(30,0), key='-FILENAME-')],
            [sg.Text('Folder:'), sg.Push()],
            [sg.InputText('', size=(30,0), key='-FOLDERPATH-')],
            [sg.Text('Path:'), sg.Push()],
            [sg.InputText('', size=(30,0), key='-FULLPATH-')],
            [sg.HorizontalSeparator(p=(0, 20))],
            [sg.FileBrowse(size=(25, 1), enable_events=True)],
            [sg.Button('Load', size=(25, 1))],
            #[sg.Push(), sg.Button("⇦",k="previous_image", size=(3, 1)), sg.Push(), sg.Button("⇨", k="next_image", size=(3,1)), sg.Push()],
            [sg.HorizontalSeparator(p=(0, 20))],
            [sg.Listbox(values=[], size=(27,30), horizontal_scroll=True, enable_events=True, key='-LISTBOX-')],
        ]

        layout = [[sg.Column(image_column),
                   sg.VerticalSeparator(),
                   sg.Column(output_column, vertical_alignment='Top', element_justification='c')
                #    sg.Column(adjustment_column, vertical_alignment='Top')
                   ]]

        self.window = sg.Window('SEM', layout, finalize=True ,return_keyboard_events=True)

        self.window['image'].bind('<B1-Motion>', 'pan')
        #self.window.bind('<Left>', 'previous_image')
        self.window.bind('<Up>', 'previous_image')
        #self.window.bind('<Right>', 'next_image')
        self.window.bind('<Down>', 'next_image')
        # self.window['image'].bind('<Button-1>', 'pan')

        # self.rgb_release = [x + ' release' for x in self.rgb_adjustments]

        # State
        self.__LOADED__ = False
        self.__DIAMETER__ = 30

        self.base_folder = None
        self.valid_image_files = []
        self.current_image_index = 0

        # Image Buffers
        self.raw_image = None  # raw image from file
        self.scaled_image = None  # to adjust cell diameter size
        self.scaled_and_adjusted_image = None

        self.display_image_scaled = None
        self.display_image = None
        self.scale_ratio = None

        self.fig_agg = None
        self.fig = None

        self.model = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    def main_loop(self):

        while True:
            event, values = self.window.read()

            if event == 'pan':
                print('CLICKED: ', values['pan'])

            if event == 'Exit' or event == sg.WIN_CLOSED:
                print('Program closed.')
                return

            # Load an image for the first time
            if event == 'Load' and values['Browse'] == '':
                sg.popup_ok('No File Selected. Please select a file via "Browse"')
                print('Error: Tried to load without selecting file.')

            if event == 'Load' and values['Browse'] != '':
                self.base_folder = os.path.split(values['Browse'])[0]
                print('==============================')
                print('FOLDER PATH:')
                print(self.base_folder)
                print('==============================')
                self.window['-FOLDERPATH-'].update(self.base_folder)
                self.valid_image_files = []
                self.valid_image_file_names = []

                for ext in ['*.png', '*.tif', '*.jpeg', '*.tiff', '*.TIF', '*.jpg']:
                    self.valid_image_files.extend(glob.glob(os.path.join(self.base_folder, ext)))

                # print('VALID IMAGE FILES:')
                # print('==============================')
                # for file in self.valid_image_files:
                #     print(file)
                # print('==============================')

                self.valid_image_files = [*set(self.valid_image_files)]
                self.valid_image_files.sort()

                for i, f in enumerate(self.valid_image_files):
                    if os.path.split(f) == os.path.split(values['Browse']):
                        self.current_image_index = i
                        break
                
                self.window['-LISTBOX-'].update(self.valid_image_files)

                self.load_image(self.valid_image_files[self.current_image_index])

                self.draw_image()

            if event == 'previous_image' and values['Browse'] != '':
                self.current_image_index = self.current_image_index - 1
                if self.current_image_index < 0:
                    self.current_image_index = len(self.valid_image_files) - 1
                try:
                    self.load_image(self.valid_image_files[self.current_image_index])
                except Exception as e:
                    print(f'An error occurred: {e}')
                    # print(self.current_image_index, len(self.valid_image_files), self.valid_image_files)

                self.draw_image()

                try:
                    self.window['-FILENAME-'].update(os.path.basename(os.path.normpath(self.valid_image_files[self.current_image_index])))
                    self.window['-LISTBOX-'].update(set_to_index=[self.current_image_index], scroll_to_index=self.current_image_index)
                except:
                    pass

            if event == 'next_image' and values['Browse'] != '':
                self.current_image_index = self.current_image_index + 1
                if self.current_image_index > len(self.valid_image_files) - 1:
                    self.current_image_index = 0

                try:
                    self.load_image(self.valid_image_files[self.current_image_index])
                except Exception as e:
                    print(f'An error has occurred: {e}')
                    # print(self.current_image_index, len(self.valid_image_files), self.valid_image_files)

                self.draw_image()
                
                try:
                    self.window['-FILENAME-'].update(os.path.basename(os.path.normpath(self.valid_image_files[self.current_image_index])))
                    self.window['-LISTBOX-'].update(set_to_index=[self.current_image_index], scroll_to_index=self.current_image_index)
                except:
                    pass
                
            elif event == '-LISTBOX-' and values['Browse'] != '':

                    f = values['-LISTBOX-'][0]
                    self.current_image_index = self.valid_image_files.index(f)
                    
                    try:
                        self.load_image(self.valid_image_files[self.current_image_index])
                    except Exception as e:
                        print(f'An error has occurred: {e}')

                    self.draw_image()
                    self.window['-FILENAME-'].update(os.path.basename(os.path.normpath(self.valid_image_files[self.current_image_index])))


    def load_image(self, f: str):

        img: np.array = hcat.lib.utils.load(f, verbose=False)

        self.window['-LOG-'].update('')

        if img is not None:

            try:

                scale: int = hcat.lib.utils.get_dtype_offset(img.dtype, img.max())
                self.raw_image: Tensor = hcat.lib.utils.image_to_float(img, scale, verbose=False).to(self.device)
                self.raw_image: Tensor = hcat.lib.utils.make_rgb(self.raw_image)  # Ensure at least 3 colors
                self.scaled_image: Tensor = hcat.lib.utils.correct_pixel_size_image(self.raw_image, None,
                                                                                    cell_diameter=float(self.__DIAMETER__),
                                                                                    verbose=False).to(self.device)

                _, x, y = self.scaled_image.shape
                self.ratio = min(900 / x, 900 / y)
                self.display_image_scaled = F.interpolate(self.scaled_image.unsqueeze(0),
                                                        scale_factor=(self.ratio, self.ratio)).squeeze(0)
                print(f'Loaded image and scaled to shape {self.display_image_scaled.shape}\nImage: {f}')
                self.window['-FULLPATH-'].update(f'{f}')
                self.window['-LOG-'].update(os.path.basename(os.path.normpath(f)))
                self.__LOADED__ = True
            
            except Exception as e:

                print(f'Error loading file: {e}\nFile: {f}')
                self.window['-FULLPATH-'].update(f'{f}')
                self.window['-LOG-'].update(f'ERROR: {e} \n{os.path.basename(os.path.normpath(f))}')

        
        else:

            print(f'Error loading file. \nFile: {f}')
            self.window['-FULLPATH-'].update(f'{f}')
            self.window['-LOG-'].update(f'ERROR: IMAGE NOT FOUND \n{os.path.basename(os.path.normpath(f))}')
            # sg.popup_ok(f'Failed to Load: \n{f}\nFile Not Found.')

        self.window['-FILENAME-'].update(os.path.basename(os.path.normpath(f)))

    def delete_fig_agg(self):
        if self.fig_agg:
            self.fig_agg.get_tk_widget().forget()
            plt.close('all')
        self.window.refresh()

    def draw_image(self):
        # cell_key = {'OHC': '#56B4E9', 'IHC': '#E69F00'}

        if not self.__LOADED__:
            return

        _image = self.display_image_scaled.clone()[0:3, ...]
        # weight = torch.ones((3, 1, 1), device=self.device) * torch.tensor(self.contrast, device=self.device).view(3, 1, 1)
        # bias = torch.ones((3, 1, 1), device=self.device) * torch.tensor(self.brightness, device=self.device).view(3, 1, 1)

        # _image = _image - (0.5 - bias * 2)
        # _image = torch.clamp(_image * weight + bias + (0.5 - bias * 2), 0, 1)
        # _image = _image * torch.tensor(self.rgb, device=self.device).view(3, 1, 1)


        # _image: Tensor = hcat.lib.utils.make_rgb(_image)
        _image: Tensor = _image.mul(255).round().to(torch.uint8).cpu()

        # if self.labels:
        #     color = [cell_key[l] for l in self.labels]
        #     _image: Tensor = torchvision.utils.draw_bounding_boxes(image=_image,
        #                                                         boxes=self.boxes,
        #                                                         colors=color)

        img = encode_png(_image, 0).numpy().tobytes()
        _, x, y = _image.shape

        self.window['image'].update(data=img, size=(y, x))
        self.display_image = _image



if __name__ == '__main__':
    gui().main_loop()
