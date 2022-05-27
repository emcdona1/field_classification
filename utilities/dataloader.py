import os
from pathlib import Path
import pickle
import requests
import cv2
from datetime import datetime
from typing import Union, List
import numpy as np
import pandas as pd
from urllib.request import urlopen, Request


def load_list_from_txt(file_path: Union[str, Path]) -> list:
    results = list()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for item in lines:
            item = item.strip()
            results.append(item)
    return results


def load_file_list_from_filesystem(directory_or_file: Union[str, Path]) -> List[Path]:
    if os.path.isdir(directory_or_file):
        directory_or_file = Path(directory_or_file)
        all_directory_contents = os.listdir(directory_or_file)
        all_directory_contents_with_full_path = [Path(directory_or_file, filename)
                                                 for filename in all_directory_contents]
        results = [item for item in all_directory_contents_with_full_path if not os.path.isdir(item)]
    elif os.path.isfile(directory_or_file):
        results = [Path(directory_or_file)]
    else:
        raise FileNotFoundError(f'Not a valid directory or file: {directory_or_file}')

    return results


def load_pickle(pickle_file_path: Union[Path, str]):
    with open(pickle_file_path, 'rb') as file:
        de_pickled = pickle.load(file)
    return de_pickled


def pickle_an_object(save_directory: Union[Path, str], object_id: str, obj_to_pickle: any) -> Path:
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    filename = object_id + '.pickle'
    file_path = Path(save_directory, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(obj_to_pickle, file)
    return file_path


def open_cv2_image(image_location: Union[str, Path, Request], is_rgb_image: bool = True) -> np.ndarray:
    color_mode = cv2.IMREAD_COLOR if is_rgb_image else cv2.IMREAD_GRAYSCALE
    if type(image_location) is Request or (type(image_location) is str and 'http' in image_location):
        resp = urlopen(image_location)
        image = np.asarray(bytearray(resp.read()), dtype='uint8')
        image_to_draw_on = cv2.imdecode(image, color_mode)
    else:
        image_to_draw_on = cv2.imread(str(image_location), color_mode)
    return image_to_draw_on


def save_cv2_image(save_location: Union[Path, str], image_id: str, image_to_save: np.ndarray) -> str:
    filename = f'{image_id}-annotated-{get_timestamp_for_file_saving()}.jpg'
    file_path = Path(save_location, filename)
    cv2.imwrite(str(file_path), image_to_save)
    return filename


def get_timestamp_for_file_saving() -> str:
    return datetime.strftime(datetime.now(), '%Y_%m_%d-%H_%M_%S')


def save_dataframe_as_csv(save_location: Union[Path, str], file_id: str, df: pd.DataFrame, timestamp=True) -> Path:
    file_location = Path(save_location, f'{file_id}{"-" + get_timestamp_for_file_saving() if timestamp else ""}.csv')
    df.to_csv(file_location, index=False, encoding='UTF-8')
    return file_location


def download_image(image_url: str, save_directory: Union[Path, str], image_id: str) -> Path:
    image_save_path = Path(save_directory, f'{image_id}.jpg')
    if os.path.exists(image_save_path):
        image_save_path = 'ALREADY_EXISTS'
    else:
        result = requests.get(image_url)
        if result.status_code == 200:
            with open(image_save_path, 'wb') as f:
                f.write(result.content)
        else:
            image_save_path = ''
    return image_save_path
