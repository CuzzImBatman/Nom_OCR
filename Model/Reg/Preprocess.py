from PIL import Image
import os
import sys
import numpy as np # linear algebra
import cv2 # image processing

from glob import glob
def make_square(im, fill_color=(255, 255, 255, 0)):
    x, y = im.size
    size = max( x, y)
    new_im =  Image.new('L', (size, size), color=255)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
target = sys.argv[1]


output_dir = f'D:/{target}_Preprocess/'
input_dir = f'D:/{target}'
os.makedirs(output_dir, exist_ok=True)

sub_folders = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
def preprocess(filepath, sub):
        for filename in filepath:
            #print(filename)
            filename = filename.split('./')[-1]
            #filename = filename.replace('.jpg', '.png')
            #print(filename)
            img = Image.open(filename)
            img= make_square(img)
            img= img.resize((40,40))
            img= img.convert('L')
            
            #prepro = apply_ben_preprocessing(img)
            #after = apply_denoising(prepro)
            filename = filename.split('\\')[-1]
            save_path = output_dir + sub + '/'+ filename
            if not os.path.exists(output_dir + sub ):
                os.makedirs(output_dir + sub )
            img=img.save(save_path)
            

#target_files = glob(f"./kuzushiji-recognition/version_{target}/*.jpg")
for name in sub_folders:
    
    target_files = glob(f'D:/{target}/{name}/*.jpg')
    #print(target_files)
    preprocess( target_files,name)
#print(target_files)
