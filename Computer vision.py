import os
import tensorflow.python as tf
import numpy as np
from tkinter import *
from tkinter import filedialog
from tensorflow.python import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageTk
from translate import Translator


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

translator = Translator(to_lang="ru")

# Загрузка предварительно обученной модели MobileNetV2
model = MobileNetV2(weights='imagenet')

tk = Tk()
tk.title('Computer vision')
canvas = Canvas(width=880, height=600, bg="white")
canvas.pack()

bg = PhotoImage(file="System_image\\fon 886x610.png")
canvas.create_image(0, 0, anchor=NW, image=bg)

fon = PhotoImage(file='System_image\\bel_fon.png')
canvas.create_image(460, 19, anchor=NW, image=fon)

icon = PhotoImage(file='System_image\\icon100x70.png')
canvas.create_image(10, 19, anchor=NW, image=icon)

ramk = PhotoImage(file='System_image\\ramka410x280.png')
canvas.create_image(25, 280, anchor=NW, image=ramk)

infwind = Label(tk, text=('Загрузите изображение,\n чтобы узнать,\n что видит ИИ на фотографии'),
                font=("Arial Bold", 20), anchor=NW, bg='white')
infwind.place(x=460, y=120)


def translate_text(text):
    text = text.replace('_', ' ')
    translation = translator.translate(text)
    return translation


def predict_image(image_path):
    global opic
    global opicru
    global count
    text = ''
    # Загрузка и предобработка изображения
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Получение предсказаний от модели
    predictions = model.predict(img_array)

    # Декодирование предсказаний
    decoded_predictions = decode_predictions(predictions)

    # Вывод топ-3 предсказаний
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        if i == 3:
            break
        else:
            text += (f"{i + 1}: {label.title()} -  ({score:.2f}) \n")
    if count == 1:
        opic.place_forget()
        opicru.place_forget()
    count = 1
    opic = Label(tk, text=(text), font=("Arial Bold", 14), anchor=E, justify="left", bg='white', )
    opic.place(x=42, y=305)
    opicru = Label(tk, text=(translate_text(text)), font=("Arial Bold", 14), anchor=E, justify="left", bg='white')
    opicru.place(x=42, y=410)


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((400, 510))
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo)
        image_label.image = photo
        predict_image(file_path)


count = 0

image_label = Label(tk, bg='white')
image_label.place(x=460, y=19)

open_button = Button(tk, text='Загрузить изображение', font=("Arial Bold", 14), fg="black", bg='white',
                     command=open_image)
open_button.place(x=562, y=547)

tk.mainloop()
