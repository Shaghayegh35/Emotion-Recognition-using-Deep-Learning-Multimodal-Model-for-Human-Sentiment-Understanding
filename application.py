# In this code we implemented a GUI application for uploading images represented emotions and 
# check the prediction of our best model(VGG16).
# In the first step, we create root using the code root = tk.Tk, and 
# then using the add_image function, which is the function of the upload_button button 
# (we call the image using the photo address that is in the system and display the desired photo in the lableshow label)
# And we show the emotion prediction text in labletext
# And by using place, the place of the button is specified on the root
# Using lable, we place the facial emotion recognition text on the page



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

# Load the pre-trained model
cur_path = os.getcwd()

optims = [optimizers.SGD(learning_rate=0.001,momentum=0.9,nesterov=True),]
model = load_model(cur_path+'/'+'model_3.h5',compile=False)
model.compile(loss = 'categorical_crossentropy',
              optimizer = optims[0],
              metrics = ['accuracy'])


root = tk.Tk()
root.title("Emotion Detection Application")
root.geometry("1700x800")
root.attributes('-fullscreen', True)
def add_image():
        file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file_path:
                image = Image.open(file_path)
                image_1=image.resize((450, 350))
                photo = ImageTk.PhotoImage(image_1)
                lableshow.config(image=photo)
                lableshow.photo = photo
                lableshow.config(text=f"Image loaded: {file_path}")
     
 
        image_array = np.array(image)
        resized_image = image_array.reshape(48,48)
        t_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

         # Make predictions
        predictions = model.predict(t_image.reshape(1,48,48,3))
        mapper = {
                0: 'anger',
                1: 'disgust',
                2: 'fear',
                3: 'happiness',
                4: 'sadness',
                5: 'surprise',
                6: 'neutral'
        }
        output=mapper[np.argmax(predictions)]
        labletext = tk.Label(root, text=output,font = ("Comic Sans MS", 50, "bold"),bg = '#1b1e24',width=7, height=2, fg="white")
        labletext.place(relx=0.7, rely=0.85, anchor="center")
        return output  
        


# Explicitly set the background color for the root window
root.configure(bg='#1b1e24')

LabelTitle = tk.Label(root, text= 'Facial Emotion Recognition',font = ("Comic Sans MS", 50, "bold"),bg = '#1b1e24', fg="white")
LabelTitle.place(relx=0.07, rely=0.15)



imagetabexample2 = ImageTk.PhotoImage(Image.open("icons8-folder-200.png"))
first_lable = tk.Label(root,bg='#1b1e24',image = imagetabexample2)
first_lable.place(relx=0.2, rely=0.57, anchor="center")

lableshow = tk.Label(root)
lableshow.place(relx=0.7, rely=0.5, anchor="center")


upload_button = tk.Button(root, text="Upload Image", padx=10, pady=5, fg="black", bg="white", command=add_image)
upload_button.place(relx=0.153, rely=0.70)


root.mainloop()
