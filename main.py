import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

window = tk.Tk()
window.title("Распознавание букв и цифр")

pixel_size = 15
is_training = True

symbol_data = []
symbol_arr = np.zeros((28, 28), dtype=np.uint8)

symbol_label = []

train_label = tk.Label(window, text="Обучение", bg='lightgreen')
train_label.grid(column=0, columnspan=3, row=0)

test_label = tk.Label(window, text="Тестирование", bg='pink')
test_label.grid(column=4, row=0)

canvas_width = 28 * pixel_size
canvas_height = 28 * pixel_size

drawing = False

canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.grid(column=3, row=2, rowspan=14)

def start_drawing(event):
    global drawing
    drawing = True

def stop_drawing(event):
    global drawing
    drawing = False

def draw_pixel(event, wiping=False):
    if drawing:
        x, y = event.x, event.y
        if 0 <= x < canvas_width and 0 <= y < canvas_height:
            global symbol_arr
            
            x1, y1 = x // pixel_size * pixel_size, y // pixel_size * pixel_size
            x2, y2 = x1 + pixel_size, y1 + pixel_size
            if not wiping:
                canvas.create_rectangle(x1, y1, x2, y2, fill="black")
                symbol_arr[y // pixel_size, x // pixel_size] = 1
            else:
                canvas.create_rectangle(x1, y1, x2, y2, fill="white")
                symbol_arr[y // pixel_size, x // pixel_size] = 0

canvas.bind("<ButtonPress-1>", lambda event: (start_drawing(event), draw_pixel(event, False)))
canvas.bind("<ButtonPress-3>", lambda event: (start_drawing(event), draw_pixel(event, True)))
canvas.bind("<ButtonRelease-1>", stop_drawing)
canvas.bind("<ButtonRelease-3>", stop_drawing)
canvas.bind("<B1-Motion>", draw_pixel)

symbol_selection_label = tk.Label(window, text="Выберите цифру:")
symbol_selection_label.grid(column=0, columnspan=3, row=1)

symbol = tk.IntVar()
symbol.set(0)

start = ord('а')
end = ord('я')
cyrillic_letters = [chr(code) for code in range(start, end + 1)]
digits = [str(num) for num in range(0, 10)]
symbols = cyrillic_letters + digits

val = 0
i, j = 0, 0
for label in symbols:
    radio_button = tk.Radiobutton(window, text=label, variable=symbol, value=val)
    radio_button.grid(column=i, row=j + 2)

    val += 1
    i += 1
    if i == 3:
        i = 0
        j += 1

def save_symbol_data():
    canvas.delete('all')

    global symbol_arr
    symbol_data.append(symbol_arr)
    symbol_arr = np.zeros((28, 28), dtype=np.uint8)

    global symbol_label
    global symbol
    symbol_label.append(symbol.get())

save_button = tk.Button(window, text="Сохранить", command=save_symbol_data)
save_button.grid(column=0, columnspan=3, row=16)

def fit():
    global symbol_data
    symbol_data = np.array(symbol_data)
    global symbol_label
    symbol_label = np.array(symbol_label)
    symbol_label = to_categorical(symbol_label, 42)

    try:
        file_data = np.load("data.npy")
        symbol_data = np.concatenate((file_data, symbol_data), axis=0)
        file_label = np.load("label.npy")
        symbol_label = np.concatenate((file_label, symbol_label), axis=0)
    except:
        pass
    np.save("data.npy", symbol_data)
    np.save("label.npy", symbol_label)

    symbol_data = symbol_data.reshape(-1, 28, 28, 1)
    
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1))) # Входной слой
    model.add(MaxPooling2D((2,2), strides=2))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(42, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    with tf.device('/CPU:0'):
        hist = model.fit(symbol_data, symbol_label, epochs=12, batch_size=32, validation_split=0.2)

    model.save("model.keras")

    symbol_data = []
    symbol_label = []

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    _, plots = plt.subplots(1, 2, figsize=(12, 5))

    plots[0].plot(epochs, loss, 'bo', label='Training loss')
    plots[0].plot(epochs, val_loss, 'b', label='Validation loss')
    plots[0].set_title('Training and validation loss')
    plots[0].set_xlabel('Epochs')
    plots[0].set_ylabel('Loss')
    plots[0].legend()

    plots[1].plot(epochs, acc, 'bo', label='Training acc')
    plots[1].plot(epochs, val_acc, 'b', label='Validation acc')
    plots[1].set_title('Training and validation accuracy')
    plots[1].set_xlabel('Epochs')
    plots[1].set_ylabel('Accuracy')
    plots[1].legend()

    plt.show()

save_button = tk.Button(window, text="Обучить", command=fit)
save_button.grid(column=0, columnspan=3, row=17)

def classify():
    global symbol_arr
    symbol_test_data = symbol_arr.reshape(1, 28, 28, 1)

    try:
        model = load_model("model.keras")
    
        pred = model.predict(symbol_test_data)
        pred = pred[0]

        max1, max_perc1 = np.argmax(pred), int(100 * np.amax(pred))
        pred[max1] = 0
        max2, max_perc2 = np.argmax(pred), int(100 * np.amax(pred))
        pred[max2] = 0
        max3, max_perc3 = np.argmax(pred), int(100 * np.amax(pred))
        pred[max3] = 0

        global symbols
        max1, max2, max3 = symbols[max1], symbols[max2], symbols[max3]

        class_label.configure(text=f"{max1} с вероятностью {max_perc1}%\n"
                                   f"{max2} с вероятностью {max_perc2}%\n"
                                   f"{max3} с вероятностью {max_perc3}%")       
    except:
        messagebox.showwarning('Предупреждение', 'Модель ещё не обучена')

classify_label = tk.Button(window, text="Классифицировать", command=classify)
classify_label.grid(column=4, row=1)

class_label = tk.Label(window, text="")
class_label.grid(column=4, row=2)

def clear():
    canvas.delete('all')

    global symbol_arr
    symbol_arr = np.zeros((28, 28), dtype=np.uint8)

    class_label.configure(text="")

clear_button = tk.Button(window, text="Очистить холст", command=clear)
clear_button.grid(column=3, row=16)

window.mainloop()
