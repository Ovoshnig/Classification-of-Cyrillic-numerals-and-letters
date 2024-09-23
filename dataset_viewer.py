import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Загрузка данных из файлов .npy
data = np.load('data.npy')
labels = np.load('label.npy')

current_index = 0
num_images = 10

# Создание областей для отображения изображений
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
plt.subplots_adjust(bottom=0.2)
plt.suptitle('Images')

# Функция для отображения следующих 10 изображений
def show_images(images, labels, index, num):
    for i in range(num):
        if index + i < len(images):
            row = i // 5
            col = i % 5
            axes[row, col].clear()
            axes[row, col].imshow(images[index + i], cmap='gray')
            #axes[row, col].set_title(f'Label: {labels[index + i]}')
            axes[row, col].axis('off')
    plt.pause(0.01)
    plt.draw()

def next_images(event):
    global current_index
    current_index += num_images
    show_images(data, labels, current_index, num_images)

# Показать первые 10 изображений из датасета
show_images(data, labels, current_index, num_images)

# Создание кнопки для отображения следующих 10 изображений
ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
btn_next = Button(ax_button, 'Next')
btn_next.on_clicked(next_images)

plt.show()
