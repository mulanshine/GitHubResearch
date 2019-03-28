import numpy as np
import scipy.misc
import PIL.Image
import matplotlib.pyplot as plt
# path = "/home/jjchu/DataSet/spinalcord/cropmask_rect/site1-sc01-mask-r2_1.pgm"
im = PIL.Image.open("/home/jjchu/DataSet/spinalcord/crop_100/image_crop/site1-sc01-mask-r1_2.pgm")
im = im.convert('L')
im_mat = scipy.misc.fromimage(im)
im_mat = im_mat/255.0
rows, cols = im_mat.shape

def get_padshape(image):
    hight = rows * 2
    weight = cols * 2 
    pad_weight1 = int((weight - image.shape[1])/2)
    pad_weight2 = weight - image.shape[1] - pad_weight1
    pad_hight1 = int((hight - image.shape[0])/2)
    pad_hight2 = hight - image.shape[0] - pad_hight1
    return pad_weight1, pad_weight2, pad_hight1, pad_hight2

im_mat_fu = np.fft.fft2(im_mat)

pad_weight1, pad_weight2, pad_hight1, pad_hight2 = get_padshape(im_mat_fu)

im_mat_fu_pad = np.pad(im_mat_fu,((pad_hight1, pad_hight2),(pad_weight1, pad_weight2)),"constant")
# im_mat_fu_pad = np.pad(im_mat_fu,((rows,0),(cols,0)),"constant")


im_converted_mat = np.fft.ifft2(im_mat_fu_pad)

im_converted_mat = np.abs(im_converted_mat)
im_converted_mat = int(im_converted_mat * 255)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()