import drjit as dr
import mitsuba
import numpy as np
import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

def numpy_iou(predict, target):
    #predict = predict[0]
    #target = target[0]
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    print(intersect.nelement())
    y = 1. - (intersect / union).sum() / intersect.nelement()
    return y

@dr.wrap_ad(source='drjit', target='torch')
def torch_add(x):
    dims = tuple(range(x.ndimension())[1:])
    return torch.sum(x, dim=dims)

def dr_jit_iou(predict, target):
    intersect = torch_add(predict * target)
    union = torch_add(predict + target - predict * target) + 1e-6
    intersect_shape_x = intersect.shape
    x = 1.0 - dr.sum(intersect / union) / intersect_shape_x[0]
    return x

# numpy + torch
filename_input_1 = r"C:\Users\Kerstin\Downloads\source.npy"
images_1 = np.load(filename_input_1).astype('float32') / 255.
images_gt = torch.from_numpy(images_1).cuda()
v = numpy_iou(images_gt[:, 3], images_gt[:, 3])

# mitsuba + dr jit
a = images_1[:, 3]
x = mi.TensorXf(a)
dr.enable_grad(x)
b = dr_jit_iou(x, x)

test = dr.zeros(dr.cuda.ArrayXf, shape=(64))
print(test.shape)
exit()
