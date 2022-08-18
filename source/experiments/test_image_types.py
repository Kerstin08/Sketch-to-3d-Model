from PIL import Image
import numpy as np
import cv2
import os
import OpenEXR
import Imath

def exr2numpy(exr_path, chanel_name):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = file.channel(chanel_name, Float_Type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return (channel)

im_frame = Image.open("..\\..\\output\\model_normalized_normal.png")
np_frame = np.array(im_frame.getdata())
with open("..\\..\\output\\model_normalized_normal.txt", 'w') as f:
    for frame in np_frame:
        f.writelines(str(frame) + "\n")

channel_names = ['X','Y','Z']
channels = []
for channel_name in channel_names:
    channel = exr2numpy("..\\..\\output\\model_normalized_normal.exr", channel_name)
    channels.append(channel)

RGB = np.dstack(channels)
with open("..\\..\\output\\model_normalized_exr.txt", 'w') as f:
    for frame in RGB:
        f.writelines(str(frame) + "\n")
print("done")

