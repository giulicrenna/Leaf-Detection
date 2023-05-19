from functions import *
from configs import *
from progress.bar import Bar

EXTENSION : str = ".png"
GROUP_PATH : list = []
SAVE_PATH : str = f"leaf_masking/{GROUP_NAME}/csv/"

watershade_mean : list = []

for file in os.listdir(f"leaf_masking/{GROUP_NAME}"):
    path : str = os.path.join(f"leaf_masking/{GROUP_NAME}", file)
    if os.path.isfile(path):
        GROUP_PATH.append(path)
        
if __name__ == '__main__':
    bar = Bar('Processing', max=len(GROUP_PATH))
    df : pd.DataFrame = pd.DataFrame()
    df['GROUP'] = [GROUP_NAME for x in range(len(GROUP_PATH))]

    standars_1 : list = []
    standars_2 : list = []
    standars_3 : list = []
    means_1 : list = []
    means_2 : list = []
    means_3 : list = []
    
    for image_path in GROUP_PATH:
        #print(f"PROCESSING: {image_path}")
        filename : str = image_path.split('/')[-1].replace(EXTENSION, "")
        masked, rgb = create_HSV_fileter(image_path)
        watershade : cv.Mat = watershade_HSV_filter(masked)
        selected_mask = cv.bitwise_and(rgb, rgb, mask=watershade)
        
        if DEBUG:
            while True:
                cv.imshow('Masked', selected_mask)
                if cv.waitKey(10) == ord('x'):
                    break
                
        std_0, std_1, std_2, mean_0, mean_1, mean_2 = calculate_std(selected_mask, filename, GROUP_NAME)
        """
        for pixel in watershade:
                    if pixel == 255:
                        accumulated += 1
                accumulated
        """
        watershade = np.asarray(watershade)
        standars_1.append(str(std_0))
        standars_2.append(str(std_1))
        standars_3.append(str(std_2))
        means_1.append(str(mean_0))
        means_2.append(str(mean_1))
        means_3.append(str(mean_2))
        watershade_mean.append(watershade_arithmetic_mean(watershade))
        
        bar.next()
    
    df['STD_RED'] = standars_1 
    df['STD_GREEN'] = standars_2
    df['STD_BLUE'] = standars_3
    df['MEAN_RED'] = means_1
    df['MEAN_GREEN'] = means_2
    df['MEAN_BLUE'] = means_3
    df['WATERSHADE_MEAN'] = watershade_mean
    
    bar.finish()
    if os.path.isdir(SAVE_PATH):
        df.to_csv(SAVE_PATH + f'{GROUP_NAME}.csv')
        print(f"PROCCESING OF {GROUP_NAME} ENDED SUCCESFULLY")
    else:
        print(f"SAVE PATH DOES NOT EXIST: {SAVE_PATH}")

#cv.imwrite("/leaf_masking/1.png", mask)