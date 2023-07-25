import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import numpy as np


cap = None
feed_mode = "color"


def convert_to_grayscale():
    global feed_mode
    feed_mode = "grayscale"


def convert_to_color():
    global feed_mode
    feed_mode = "color"

def convert_to_cartoon():
    global feed_mode
    feed_mode = "cartoon"

def convert_to_censored():
    global feed_mode
    feed_mode = "censored"

def convert_to_dither():
    global feed_mode
    feed_mode = "dither"

def convert_to_laplacian():
    global feed_mode
    feed_mode = "laplacian"

def update_feed():
    global original_frame, cap, feed_mode

    ret, frame = cap.read()

    if ret:
        original_frame = frame.copy()

        if feed_mode == "grayscale":
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(gray_frame)
        elif feed_mode == "color":
            color_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(color_frame)
        elif feed_mode == "cartoon":

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            smooth = cv2.bilateralFilter(gray, d=8, sigmaColor=75, sigmaSpace=75)


            edges = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)


            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


            edges_inv = cv2.bitwise_not(edges)


            cartoon = cv2.bitwise_and(frame, edges)


            cartoon = cv2.bilateralFilter(cartoon, d=9, sigmaColor=75, sigmaSpace=75)

            cartoon_rgb = cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY)

            img = Image.fromarray(cartoon_rgb)

        elif feed_mode == "censored":



            resized_frame = cv2.resize(original_frame, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)


            quantized_frame = cv2.resize(resized_frame, (original_frame.shape[1], original_frame.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)

            img = Image.fromarray(cv2.cvtColor(quantized_frame, cv2.COLOR_BGR2RGB))

        elif feed_mode == "dither":


            img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)


            img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)



            canny_edges = cv2.Canny(img_gray_blur, 10, 150)


            ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
            img = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

        elif feed_mode == "laplacian":

            img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)


            laplacian = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=3, scale=8, delta=0)


            laplacian_8bit = cv2.convertScaleAbs(laplacian)

            img = Image.fromarray(cv2.cvtColor(laplacian_8bit, cv2.COLOR_BGR2RGB))

        img_tk = ImageTk.PhotoImage(image=img)
        label.img_tk = img_tk
        label.config(image=img_tk)


    desired_frame_rate = 30
    delay_in_milliseconds = int(1000 / desired_frame_rate)

    root.after(delay_in_milliseconds, update_feed)


def main():
    global cap, root, label, original_frame

    root = tk.Tk()
    root.title("Camera Feed")

    label = tk.Label(root)
    label.pack(side="left")

    # Place buttons on the right side
    button_frame = tk.Frame(root)
    button_frame.pack(side="right")

    cap = cv2.VideoCapture(0)
    original_frame = None

    grayscale_button = tk.Button(button_frame, text="Grayscale", command=convert_to_grayscale)
    grayscale_button.pack(pady=5)

    color_button = tk.Button(button_frame, text="Color", command=convert_to_color)
    color_button.pack(pady=5)

    cartoon_button = tk.Button(button_frame, text="Cartoon", command=convert_to_cartoon)
    cartoon_button.pack(pady=5)

    colors_button = tk.Button(button_frame, text="128-bit Colors", command=convert_to_censored)
    colors_button.pack(pady=5)

    colors_button = tk.Button(button_frame, text="Dither", command=convert_to_dither)
    colors_button.pack(pady=5)

    colors_button = tk.Button(button_frame, text="Laplacian", command=convert_to_laplacian)
    colors_button.pack(pady=5)

    update_feed()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
