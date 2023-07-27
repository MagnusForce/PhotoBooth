import cv2
import tkinter as tk
from PIL import Image, ImageTk, ImageOps
import numpy as np

cap = None
feed_mode = "color"
fxsk = 0.01
fysk = 0.01
scale_fx = None
lap_sc = 1
scale_lap = None
sym_sc = 6
scale_sym = None
diter = 20
scale_dit = None
current_frame = None
SAVE_GIF_EVERY = 150
TOTAL_FRAMES = 5

WIDTH, HEIGHT = 640, 480
cell_width, cell_height = 12, 12
new_width, new_height = int(WIDTH / cell_width), int(HEIGHT / cell_height)

font = cv2.FONT_HERSHEY_SIMPLEX

frames_for_gif = []
frame_count = 0


def convert_to_color():
    global feed_mode
    feed_mode = "color"


def convert_to_censored():
    global feed_mode
    feed_mode = "censored"


def convert_to_dither():
    global feed_mode
    feed_mode = "dither"


def convert_to_laplacian():
    global feed_mode
    feed_mode = "laplacian"


def convert_to_symbol():
    global feed_mode
    feed_mode = "symbol"


def update_feed():
    global original_frame, cap, feed_mode, black_window, current_frame, frame_count, frames_for_gif

    ret, frame = cap.read()

    if ret:
        original_frame = frame.copy()

        if feed_mode == "color":
            color_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(color_frame)

        elif feed_mode == "censored":
            resized_frame = cv2.resize(original_frame, None, fx=fxsk, fy=fysk, interpolation=cv2.INTER_NEAREST)
            quantized_frame = cv2.resize(resized_frame, (original_frame.shape[1], original_frame.shape[0]),
                                         interpolation=cv2.INTER_NEAREST)
            img = Image.fromarray(cv2.cvtColor(quantized_frame, cv2.COLOR_BGR2RGB))

        elif feed_mode == "dither":
            img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
            canny_edges = cv2.Canny(img_gray_blur, 20, diter)
            ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
            img = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

        elif feed_mode == "laplacian":
            img_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=1, scale=lap_sc, delta=0)
            laplacian_8bit = cv2.convertScaleAbs(laplacian)
            img = Image.fromarray(cv2.cvtColor(laplacian_8bit, cv2.COLOR_BGR2RGB))

        elif feed_mode == "symbol":
            global black_window
            black_window = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            small_image = cv2.resize(original_frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            for i in range(new_height):
                for j in range(new_width):
                    color = small_image[i, j]
                    B = int(color[0])
                    G = int(color[1])
                    R = int(color[2])
                    coord = (j * cell_width + cell_width // 2, i * cell_height + cell_height // 2)
                    if toggle_var.get():
                        img = cv2.circle(black_window, coord, sym_sc, (B, G, R), 2)
                    else:
                        x1, y1 = coord[0] - sym_sc // 2, coord[1] - sym_sc // 2
                        x2, y2 = coord[0] + sym_sc // 2, coord[1] + sym_sc // 2
                        img = cv2.rectangle(black_window, (x1, y1), (x2, y2), (B, G, R), -1)

            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        current_frame = img

        frame_count += 1

        if frame_count % SAVE_GIF_EVERY == 0 and len(frames_for_gif) < TOTAL_FRAMES:
            current_frame_np = np.array(img)
            frames_for_gif.append(current_frame_np)
            print("appended")

        # Limit the number of frames to 10
        if len(frames_for_gif) == TOTAL_FRAMES:
            save_gif()
            frames_for_gif = []

        img_tk = ImageTk.PhotoImage(image=img)
        label.img_tk = img_tk
        label.config(image=img_tk)

    desired_frame_rate = 30
    delay_in_milliseconds = int(1000 / desired_frame_rate)

    root.after(delay_in_milliseconds, update_feed)


def update_censored_scale(val):
    global fxsk, fysk
    fxsk = fysk = float(scale_fx.get())


def update_lap_scale(val):
    global lap_sc
    lap_sc = float(scale_lap.get())


def update_sym_scale(val):
    global sym_sc
    sym_sc = int(scale_sym.get())


def update_dit_scale(val):
    global diter
    diter = int(scale_dit.get())


def toggle():
    toggle_var.set(not toggle_var.get())
    if toggle_var.get():
        toggle_button.config(text="Round")
    else:
        toggle_button.config(text="Square")


def save_current_frame():
    global current_frame
    if current_frame is not None:
        current_frame.save("current_frame.jpg")


def save_gif():
    global frames_for_gif
    if len(frames_for_gif) >= TOTAL_FRAMES:
        gif_frames = [Image.fromarray(frame) for frame in frames_for_gif]
        gif_frames[0].save("every_10_frames.gif", save_all=True, append_images=gif_frames[1:], duration=500, loop=0)
        print("GIF saved successfully.")



def main():
    global cap, root, label, original_frame, scale_fx, scale_fy, scale_lap, WIDTH, HEIGHT, scale_sym, scale_dit, \
        toggle_var, toggle_button, current_frame, frames_for_gif

    root = tk.Tk()
    root.title("Camera Feed")

    root.resizable(width=False, height=False)

    label = tk.Label(root)
    label.pack(side="left")

    # Frame for the buttons on the right side
    button_frame = tk.Frame(root)
    button_frame.pack(side="right", padx=5, pady=5)

    cap = cv2.VideoCapture(0)
    original_frame = None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    toggle_var = tk.BooleanVar(value=False)

    color_button = tk.Button(button_frame, text="Color", command=convert_to_color)
    color_button.pack(pady=2, ipadx=35)

    scale_frame = tk.Frame(button_frame, bd=2, relief=tk.RAISED)
    scale_frame.pack(pady=2)

    colors_button = tk.Button(scale_frame, text="128-bit Colors", command=convert_to_censored)
    colors_button.pack(pady=2)

    # Scale widget
    scale_fx = tk.Scale(scale_frame, from_=0.01, to=0.2, resolution=0.01, orient=tk.HORIZONTAL,
                        command=update_censored_scale)
    scale_fx.set(fxsk)
    scale_fx.pack()

    dit_frame = tk.Frame(button_frame, bd=2, relief=tk.RAISED)
    dit_frame.pack(pady=2)

    colors_button = tk.Button(dit_frame, text="Dither", command=convert_to_dither)
    colors_button.pack(pady=2)

    scale_dit = tk.Scale(dit_frame, from_=20, to=100, resolution=10, orient=tk.HORIZONTAL,
                         command=update_dit_scale)
    scale_dit.set(diter)
    scale_dit.pack()

    # Frame for the scale
    lap_frame = tk.Frame(button_frame, bd=2, relief=tk.RAISED)
    lap_frame.pack(pady=2)

    colors_button = tk.Button(lap_frame, text="Laplacian", command=convert_to_laplacian)
    colors_button.pack(pady=2)

    scale_lap = tk.Scale(lap_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
                         command=update_lap_scale)
    scale_lap.set(lap_sc)
    scale_lap.pack()

    sym_frame = tk.Frame(button_frame, bd=2, relief=tk.RAISED)
    sym_frame.pack(pady=2)

    colors_button = tk.Button(sym_frame, text="Symbol", command=convert_to_symbol)
    colors_button.pack(pady=2)

    scale_sym = tk.Scale(sym_frame, from_=1, to=40, resolution=1, orient=tk.HORIZONTAL,
                         command=update_sym_scale)
    scale_sym.set(sym_sc)
    scale_sym.pack()

    toggle_button = tk.Button(sym_frame, text="Square", command=toggle)
    toggle_button.pack(pady=2)

    save_button = tk.Button(button_frame, text="Save Frame", command=save_current_frame)
    save_button.pack(pady=2, ipadx=19)

    save_button = tk.Button(button_frame, text="Save Gif", command=save_gif)
    save_button.pack(pady=2, ipadx=19)

    update_feed()

    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
