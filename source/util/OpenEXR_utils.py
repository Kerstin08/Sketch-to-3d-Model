import OpenEXR
import Imath
import numpy as np

def exr2numpy(exr_path, chanel_name):
    file = OpenEXR.InputFile(exr_path)
    header = file.header()
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = file.channel(chanel_name, Float_Type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)
    return (channel)


def getRGBimageEXR(path):
    # RGB, although data is technically xyz, however due to the conversion to vector this is RGB
    channel_names = ['R', 'G', 'B']
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

def writeDepthImage(image, path):
    img = np.squeeze(image)
    size = img.shape
    exrHeader = OpenEXR.Header(size[1], size[0])

    exrHeader['channels'] = {"T": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1)}
    exrOut = OpenEXR.OutputFile(path, exrHeader)
    GRAY = (img[:, :]).astype(np.float32).tobytes()
    exrOut.writePixels({'T': GRAY})
    exrOut.close()

def writeRGBImage(image, path):
    img = image.numpy().squeeze()
    size = img.shape
    header = OpenEXR.Header(size[1], size[2])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(path, header)
    R = (img[0, :, :]).astype(np.float16).tostring()
    G = (img[1, :, :]).astype(np.float16).tostring()
    B = (img[2, :, :]).astype(np.float16).tostring()
    out.writePixels({'R': R, 'G': G, 'B': B})
    out.close()
