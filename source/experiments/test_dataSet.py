import source.map_generation.dataset_generation.DataSet as DataSet
import matplotlib.pyplot as plt

epochs = 10
input_dir = "..\\..\\resources\\n_meshes"
target_dir = "..\\..\\resources\\sketch_meshes"
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
dataSet = DataSet.DS(target_dir, input_dir)
for idx, data in enumerate(dataSet):
    figure.add_subplot(rows, cols, (idx*2+1))
    plt.axis("off")
    plt.imshow(data['input'])
    figure.add_subplot(rows, cols, (idx*2+2))
    plt.axis("off")
    plt.imshow(data['target'], cmap="gray")
plt.show()