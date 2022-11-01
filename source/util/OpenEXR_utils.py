import OpenEXR
import Imath
import numpy as np
import source.util.data_type as data_type

def exr2numpy(exr_path, chanel_name):
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = file.channel(chanel_name, Float_Type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return (channel)


def getRGBimageEXR(path, given_data_type, axis):
    # RGB, although data is technically xyz, however due to the conversion to vector this is RGB
    if given_data_type == data_type.Type.normal:
        channel_names = ['R', 'G', 'B']
    else:
        channel_names = ['R']
    channels = []
    for channel_name in channel_names:
        channel = exr2numpy(path, channel_name)
        channels.append(channel)
    RGB = np.stack(channels, axis=axis)
    return RGB

def writeRGBImage(image, given_data_type, path):
    img = image.detach().cpu().numpy().squeeze()
    size = img.shape
    if given_data_type == data_type.Type.normal:
        header = OpenEXR.Header(size[1], size[2])
    else:
        header = OpenEXR.Header(size[0], size[1])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(path, header)
    if given_data_type == data_type.Type.normal:
        R = (img[0, :, :]).astype(np.float16).tostring()
        G = (img[1, :, :]).astype(np.float16).tostring()
        B = (img[2, :, :]).astype(np.float16).tostring()
    else:
        R = (img[:, :]).astype(np.float16).tostring()
        G = (img[:, :]).astype(np.float16).tostring()
        B = (img[:, :]).astype(np.float16).tostring()
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()
