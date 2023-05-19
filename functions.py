import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from configs import *

def calculate_std(img : cv.Mat, filename : str = "file", GROUP_NAME_ : str = "GROUP_1") -> tuple[int, int, int]:
    df : pd.DataFrame = pd.DataFrame()
    columns = ['red', 'green', 'blue']
    std : list = []
    mean : list = []
    for component in range(0, 3):
        color_arr : list = []
        for row in img:
            for pixel in row:
                color_arr.append(pixel[component])
        df[columns[component]] = pd.Series(color_arr)
    
    for col in columns:
        df_temp = df.copy()
        df_temp = df_temp[df_temp[col] != 0]
        plt.figure(figsize=(100, 5));
        sns.countplot(data = df_temp, x = col)          
        path : str = f"leaf_masking/{GROUP_NAME_}/count_plot/"
        file_path : str = f"leaf_masking/{GROUP_NAME_}/count_plot/{col}_plot_{filename}.png"
        if os.path.isdir(path):
            plt.savefig(file_path)
        else:
            os.mkdir(path)
            plt.savefig(file_path)
        plt.close('all')
        
        dev : int = df_temp[col].std()  
        mean_ : int = df_temp[col].mean()
        std.append(dev)
        mean.append(mean_)
    
    return std[0], std[1], std[2], mean[0], mean[1], mean[2]

def create_HSV_fileter(path : str) -> tuple[cv.Mat, cv.Mat, cv.Mat]:
    #print(f"CREATING HSV FILTER FOR {path}")
    matrix_rgb : cv.Mat = cv.imread(path)
    matrix_hsv : cv.Mat = cv.cvtColor(matrix_rgb, cv.COLOR_BGR2HSV_FULL)

    lower_threshold = np.array([lower_h,
                                lower_s,
                                lower_v])
    upper_threshold = np.array([upper_h,
                                upper_s,
                                upper_v])

    mask : cv.Mat = cv.inRange(matrix_hsv,
                            lower_threshold,
                            upper_threshold)

    width : int = int(mask.shape[1] * scale_percent / 100)
    height : int = int(mask.shape[0] * scale_percent / 100)
    dim : tuple = (width, height)

    resized_rgb = cv.resize(matrix_rgb,
                            dim,
                            interpolation = cv.INTER_AREA)
    resized_mask = cv.resize(mask,
                             dim,
                             interpolation = cv.INTER_AREA)
    
    return resized_mask, resized_rgb

def watershade_HSV_filter(img : cv.Mat) -> cv.Mat:
    watershade : cv.Mat = cv.threshold(img,
                                       watershade_lower_threshold,
                                       watershade_upper_threshold,
                                       cv.THRESH_BINARY)[1]
    return watershade


def watershade_arithmetic_mean(array) -> float:
    acumulated :  list = []
    array = np.asarray(array)
    shape = array.shape
    
    for row in array:
        temp_count : int = 0
        for item in row:
            if item == 255:
                temp_count += 1
        mean : int = temp_count / shape[0]
        acumulated.append(mean)
        
    return sum(acumulated)/shape[1]