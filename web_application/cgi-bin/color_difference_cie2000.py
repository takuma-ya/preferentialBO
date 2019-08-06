from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import color_enhance
import numpy as np
import PIL.Image
from PIL import ImageEnhance

IMG_DIR = "../img/slider/"

def calculate_difference_from_file(input_file, output_file, original_img, reference_img):
    params = []
    with open(input_file) as f:
        lines=f.readlines()
        for line in lines:
            line = line.strip('[]\n ')
            param = np.array([float(i) for i in line.split(' ') if i])
            params.append(param)
    image_params = 1 + np.array(params)
    i = 0
    color_enhance.color_enhance(original_img, IMG_DIR, image_params)

    orig_img = PIL.Image.open(reference_img)
    rgb_orig = orig_img.convert('RGB')
    size = rgb_orig.size

    for i in range(10):
        img = PIL.Image.open(IMG_DIR+str(i)+".jpg")

        #RGBに変換
        rgb_img = img.convert('RGB')

        #画像サイズを取得
        size = rgb_img.size

        delta_e_list = []
        for x in range(size[0]):
            for y in range(size[1]):
                #ピクセルを取得
                r,g,b = rgb_img.getpixel((x,y))
                color_rgb = sRGBColor(r, g, b)
                color_lab = convert_color(color_rgb, LabColor)

                r,g,b = rgb_orig.getpixel((x,y))
                color_orig_rgb = sRGBColor(r, g, b)
                color_orig_lab = convert_color(color_orig_rgb, LabColor)

                delta_e_list.append(delta_e_cie2000(color_lab, color_orig_lab))

        dif = np.mean(delta_e_list)
        with open(output_file,'a') as f:
            f.write(str(dif).replace(' ','')+'\n')

if __name__ == '__main__':
    calculate_difference_from_file("../log/selected_param.txt", "../log/residual_cie2000.txt", "../img/original/s_original.jpg", "../img/reference/reference.jpg") 
