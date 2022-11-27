import mitsuba
import mitsuba as mi
import numpy as np
mi.set_variant('cuda_ad_rgb')
import OpenEXR

depth = r"C:\Users\Kerstin\Documents\MasterThesis\masterthesis_hofer_kerstin\output\32770_depth.exr"
normal = r"C:\Users\Kerstin\Documents\MasterThesis\masterthesis_hofer_kerstin\output\32770_normal.exr"
file = OpenEXR.InputFile(depth)
dw = file.header()['channels']
print(dw)
#with open("..\\..\\output\\depth.txt", "w") as f:
#    for i in range(len(temp)):
#            for j in range(len(temp[i])):
#                y = temp[i][j]
#                f.write(str(y) + "\n")
file = OpenEXR.InputFile(normal)
dw = file.header()['channels']
print(dw)
#with open("..\\..\\output\\normal.txt", "w") as f:
#    for i in range(len(temp)):
#            for j in range(len(temp[i])):
#                y = temp[i][j]
#                f.write(str(y) + "\n")
print("Done")