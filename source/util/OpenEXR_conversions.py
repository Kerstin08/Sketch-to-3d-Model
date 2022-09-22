import OpenEXR
import Imath
import numpy as np

def exr2numpy(exr_path, chanel_name):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = file.channel(chanel_name, Float_Type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return (channel)


def getRGBimageEXR(path):
    channel_names = ['X', 'Y', 'Z']
    channels = []
    for channel_name in channel_names:
        channel = exr2numpy(path, channel_name)
        channels.append(channel)
    RGB = np.stack(channels, axis=0)
    return RGB

def getDepthimageEXR(path):
    channel_names = 'T'
    channel = exr2numpy(path, channel_names)
    return np.copy(channel)