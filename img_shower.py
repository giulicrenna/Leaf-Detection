from functions import *

EXTENSION : str = ".png"
GROUP_NAME : str = "GRUPO_6"
GROUP_PATH : list = []
SAVE_PATH : str = f"leaf_masking/{GROUP_NAME}/"

DEBUG : bool = False
SHOW_IMG : bool = True

for file in os.listdir(f"leaf_masking/{GROUP_NAME}"):
    path : str = os.path.join(f"leaf_masking/{GROUP_NAME}", file)
    if os.path.isfile(path):
        GROUP_PATH.append(path)

if __name__ == '__main__':
    print(f"CREATING DATASET FOR: {GROUP_NAME}")

    if SHOW_IMG:
        for image_path in GROUP_PATH:
            print(f"PROCESSING: {image_path}")
            filename : str = image_path.split('/')[-1].replace(EXTENSION, "")
            masked, rgb = create_HSV_fileter(image_path)
            watershade : cv.Mat = watershade_HSV_filter(masked)
            selected_mask = cv.bitwise_and(rgb, rgb, mask=watershade)
            
            while True:
                cv.imshow('Masked', masked)
                cv.imshow('Original', selected_mask)
                cv.imshow('watershade', watershade)
                
                if cv.waitKey(10) == ord('x'):
                    break
#cv.imwrite("/leaf_masking/1.png", mask)