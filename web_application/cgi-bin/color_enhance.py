import numpy as np
import PIL.Image
from PIL import ImageEnhance

def color_enhance(input_file, output_dir, image_params):
    orig_img = PIL.Image.open(input_file)
    for i,j in enumerate(image_params):
        # 彩度を変える
        saturation_converter = ImageEnhance.Color(orig_img)
        img = saturation_converter.enhance(j[0])

        # コントラストを変える
        contrast_converter = ImageEnhance.Contrast(img)
        img = contrast_converter.enhance(j[1])

        # 明度を変える
        brightness_converter = ImageEnhance.Brightness(img)
        img = brightness_converter.enhance(j[2])

        # シャープネスを変える
        #sharpness_converter = ImageEnhance.Sharpness(img)
        #img = sharpness_converter.enhance(l[2])

        #　画像を保存
        img.save(output_dir+str(i)+".jpg")

if __name__ == '__main__': 
    image_params = np.tile(np.linspace(0.5,1.5,30), (3,1)).T
    color_enhance("../img/original/s_original.jpg", "../img/slider/", image_params) 
