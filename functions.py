import cv2 as cv
import seaborn as sbs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

lower_h : int = 11
lower_s : int = 16
lower_v : int = 5

upper_h : int = 62
upper_s : int = 255
upper_v : int = 203

lower_tuple : str = f"({lower_h}, {lower_s}, {lower_v})"
upper_tuple : str = f"({upper_h}, {upper_s}, {upper_v})"

scale_percent = 30

watershade_lower_threshold : int = 0
watershade_upper_threshold : int = 255

def calculate_std(img : cv.Mat, filename : str = "file", GROUP_NAME : str = "GROUP_1") -> tuple[int, int, int]:
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
        sbs.countplot(data = df_temp, x = col)          
        path : str = f"leaf_masking/{GROUP_NAME}/count_plot/"
        file_path : str = f"leaf_masking/{GROUP_NAME}/count_plot/{col}_plot_{filename}.png"
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
    print(f"CREATING HSV FILTER FOR {path}")
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
