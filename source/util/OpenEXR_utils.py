# Utils to work with OpenEXR files
import OpenEXR
import Imath
import numpy
import numpy as np
import torch

import source.util.data_type as data_type


def exr2numpy(
        exr_path: str,
        chanel_name: str
) -> numpy.ndarray:
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = file.channel(chanel_name, Float_Type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return channel


def getImageEXR(
        path: str,
        given_data_type: data_type.Type,
        axis: int
) -> numpy.ndarray:
    if given_data_type == data_type.Type.normal:
        channel_names = ['R', 'G', 'B']
    else:
        channel_names = ['Y']
    channels = []
    for channel_name in channel_names:
        channel = exr2numpy(path, channel_name)
        channels.append(channel)
    image = np.stack(channels, axis=axis)
    array_sum = np.sum(image)
    if np.isnan(array_sum):
        raise Exception("{} contains nan!".format(path))
    if np.isinf(array_sum):
        raise Exception("{} contains inf!".format(path))
    return image


def writeImage(
        image: np.ndarray | torch.Tensor,
        given_data_type: data_type.Type,
        path: str
):
    if torch.is_tensor(image):
        img = image.detach().cpu().numpy().squeeze()
    elif type(image).__module__ == np.__name__:
        img = image
    else:
        raise Exception("Image to write is neither torch tensor nor numpy array.")

    size = img.shape
    header = OpenEXR.Header(size[0], size[1])

    if given_data_type == data_type.Type.normal:
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(path, header)
        R = (img[:, :, 0]).astype(np.float16).tostring()
        G = (img[:, :, 1]).astype(np.float16).tostring()
        B = (img[:, :, 2]).astype(np.float16).tostring()
        out.writePixels({'R': R, 'G': G, 'B': B})
    else:
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "Y"])
        out = OpenEXR.OutputFile(path, header)
        R = (img[:, :]).astype(np.float16).tostring()
        out.writePixels({'Y': R})
        out.close()
