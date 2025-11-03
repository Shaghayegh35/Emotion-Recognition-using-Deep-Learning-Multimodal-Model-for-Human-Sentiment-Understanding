import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from PIL import Image
import numpy as nps
import os
import cv2
from tensorflow.keras.applications.vgg19 import preprocess_input

# Load the pre-trained model
cur_path = os.getcwd()

optims = [optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True),]
model = load_model(cur_path+'/'+'mode_SGD.h5',compile=False)
model.compile(loss = 'categorical_crossentropy',
              optimizer = optims[0],
              metrics = ['accuracy'])


root = tk.Tk()
root.title("Emotion Detection Application")
root.geometry("1700x800")
root.attributes('-fullscreen', True)


def add_image():
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico *.heic")])
    if file_path:
        image = Image.open(file_path)
        image_display = image.resize((400, 400))
        image_resized = image.resize((48, 48))  # Resize to match the input size of your model
        photo = ImageTk.PhotoImage(image_display)
        lableshow.config(image=photo)
        lableshow.photo = photo
        lableshow.config(text=f"Image loaded: {file_path}")

        # Convert image to array and preproces
        image_array = np.array(image_resized)
        
        # Ensure the image has 3 channels (for RGB)
        if image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]
        
        # Expand dimensions to match the input shape of your model
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make predictions
        predictions = model.predict(image_array)
        
        mapper = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'happiness',
            4: 'sadness',
            5: 'surprise',
            6: 'neutral'
        }
        
        output = mapper[np.argmax(predictions)]
        labletext = tk.Label(root, text=output, font=("Comic Sans MS", 50, "bold"), bg='#1b1e24', width=7, height=1)
        labletext.place(relx=0.7, rely=0.8, anchor="center")
        return output

        


# Explicitly set the background color for the root window
root.configure(bg='#1b1e24')

LabelTitle = tk.Label(root, text= 'Facial Emotion Recognition',font = ("Comic Sans MS", 50, "bold"),bg = '#1b1e24')
LabelTitle.place(relx=0.07, rely=0.15)



imagetabexample2 = ImageTk.PhotoImage(Image.open("icons8-folder-200.png"))
first_lable = tk.Label(root,bg='#1b1e24',image = imagetabexample2)
first_lable.place(relx=0.2, rely=0.57, anchor="center")

lableshow = tk.Label(root)
lableshow.place(relx=0.7, rely=0.5, anchor="center")


upload_button = tk.Button(root, text="Upload Image", padx=10, pady=5, fg="black", bg="red", command=add_image)
upload_button.place(relx=0.153, rely=0.66)


root.mainloop()
