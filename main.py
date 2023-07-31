import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class CameraApp:
    def __init__(self, width, height):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.feed_mode = "color"
        self.fxsk = 0.01
        self.fysk = 0.01
        self.lap_sc = 1
        self.sym_sc = 6
        self.diter = 20
        self.desired_frame_rate = 30

        self.WIDTH, self.HEIGHT = width, height
        self.cell_width, self.cell_height = 12, 12
        self.new_width, self.new_height = int(width / self.cell_width), int(height / self.cell_height)

        self.root = tk.Tk()
        self.root.title("Camera Feed")
        self.root.resizable(width=False, height=False)

        self.label = tk.Label(self.root)
        self.label.pack(side="left")

        self.toggle_var = tk.BooleanVar(value=False)

        color_button = tk.Button(self.root, text="Color", command=self.convert_to_color)
        color_button.pack(pady=2, ipadx=35)

        scale_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        scale_frame.pack(pady=2)

        censored_button = tk.Button(scale_frame, text="Censored", command=self.convert_to_censored)
        censored_button.pack(pady=2)

        self.scale_fx = tk.Scale(scale_frame, from_=0.01, to=0.2, resolution=0.01, orient=tk.HORIZONTAL,
                                 command=self.update_censored_scale)
        self.scale_fx.set(self.fxsk)
        self.scale_fx.pack()

        dit_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        dit_frame.pack(pady=2)

        dither_button = tk.Button(dit_frame, text="Dither", command=self.convert_to_dither)
        dither_button.pack(pady=2)

        self.scale_dit = tk.Scale(dit_frame, from_=20, to=100, resolution=10, orient=tk.HORIZONTAL,
                                  command=self.update_dit_scale)
        self.scale_dit.set(self.diter)
        self.scale_dit.pack()

        lap_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        lap_frame.pack(pady=2)

        lap_button = tk.Button(lap_frame, text="Laplacian", command=self.convert_to_laplacian)
        lap_button.pack(pady=2)

        self.scale_lap = tk.Scale(lap_frame, from_=1, to=10, resolution=1, orient=tk.HORIZONTAL,
                                  command=self.update_lap_scale)
        self.scale_lap.set(self.lap_sc)
        self.scale_lap.pack()

        sym_frame = tk.Frame(self.root, bd=2, relief=tk.RAISED)
        sym_frame.pack(pady=2)

        symbol_button = tk.Button(sym_frame, text="Symbol", command=self.convert_to_symbol)
        symbol_button.pack(pady=2)

        self.scale_sym = tk.Scale(sym_frame, from_=1, to=40, resolution=1, orient=tk.HORIZONTAL,
                                  command=self.update_sym_scale)
        self.scale_sym.set(self.sym_sc)
        self.scale_sym.pack()

        self.toggle_button = tk.Button(sym_frame, text="Square", command=self.toggle)
        self.toggle_button.pack(pady=2)

        save_jpg_button = tk.Button(self.root, text="Save Frame", command=self.save_current_frame)
        save_jpg_button.pack(pady=2, ipadx=19)

        save_gif_button = tk.Button(self.root, text="Save Gif", command=self.start_gif_making)
        save_gif_button.pack(pady=2, ipadx=28)

        self.alert_label = tk.Label(self.root, text="No messages")
        self.alert_label.pack(pady=5, padx=5)

        self.current_frame = None
        self.frames_for_gif = []
        self.frame_count = 0
        self.gif_state = 0
        self.appended_frames_count = 0

    def start(self):
        self.update_feed()
        self.root.mainloop()

    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            if self.feed_mode == "color":
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = Image.fromarray(color_frame)

            elif self.feed_mode == "censored":
                resized_frame = cv2.resize(frame, None, fx=self.fxsk, fy=self.fysk, interpolation=cv2.INTER_NEAREST)
                quantized_frame = cv2.resize(resized_frame, (frame.shape[1], frame.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
                self.current_frame = Image.fromarray(cv2.cvtColor(quantized_frame, cv2.COLOR_BGR2RGB))

            elif self.feed_mode == "dither":
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
                canny_edges = cv2.Canny(img_gray_blur, 20, self.diter)
                ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
                self.current_frame = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

            elif self.feed_mode == "laplacian":
                img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=1, scale=self.lap_sc, delta=0)
                laplacian_8bit = cv2.convertScaleAbs(laplacian)
                self.current_frame = Image.fromarray(cv2.cvtColor(laplacian_8bit, cv2.COLOR_BGR2RGB))

            elif self.feed_mode == "symbol":
                black_window = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                small_image = cv2.resize(frame, (self.new_width, self.new_height), interpolation=cv2.INTER_NEAREST)
                for i in range(self.new_height):
                    for j in range(self.new_width):
                        color = small_image[i, j]
                        B = int(color[0])
                        G = int(color[1])
                        R = int(color[2])
                        coord = (j * self.cell_width + self.cell_width // 2, i * self.cell_height + self.cell_height // 2)
                        if self.toggle_var.get():
                            cv2.circle(black_window, coord, self.sym_sc, (B, G, R), 2)
                        else:
                            x1, y1 = coord[0] - self.sym_sc // 2, coord[1] - self.sym_sc // 2
                            x2, y2 = coord[0] + self.sym_sc // 2, coord[1] + self.sym_sc // 2
                            cv2.rectangle(black_window, (x1, y1), (x2, y2), (B, G, R), -1)
                self.current_frame = Image.fromarray(cv2.cvtColor(black_window, cv2.COLOR_BGR2RGB))

            if self.gif_state == 1:
                self.frame_count += 1
                if self.frame_count % SAVE_GIF_EVERY == 0 and len(self.frames_for_gif) < TOTAL_FRAMES:
                    current_frame_np = np.array(self.current_frame)
                    self.frames_for_gif.append(current_frame_np)
                    self.appended_frames_count += 1
                    alert_text = f"Frame {str(self.appended_frames_count)} added"
                    self.alert_label.config(text=alert_text, fg="orange")
                if len(self.frames_for_gif) == TOTAL_FRAMES:
                    self.alert_label.config(text="GIF was saved", fg="green")
                    self.save_gif()
                    self.frames_for_gif = []
                    self.gif_state = 0
                    self.appended_frames_count = 0

            img_tk = ImageTk.PhotoImage(image=self.current_frame)
            self.label.img_tk = img_tk
            self.label.config(image=img_tk)

        delay_in_milliseconds = int(1000 / self.desired_frame_rate)
        self.root.after(delay_in_milliseconds, self.update_feed)

    def convert_to_color(self):
        self.feed_mode = "color"

    def convert_to_censored(self):
        self.feed_mode = "censored"

    def convert_to_dither(self):
        self.feed_mode = "dither"

    def convert_to_laplacian(self):
        self.feed_mode = "laplacian"

    def convert_to_symbol(self):
        self.feed_mode = "symbol"

    def update_censored_scale(self, val):
        self.fxsk = self.fysk = float(val)

    def update_lap_scale(self, val):
        self.lap_sc = float(val)

    def update_sym_scale(self, val):
        self.sym_sc = int(val)

    def update_dit_scale(self, val):
        self.diter = int(val)

    def toggle(self):
        self.toggle_var.set(not self.toggle_var.get())
        if self.toggle_var.get():
            self.toggle_button.config(text="Round")
        else:
            self.toggle_button.config(text="Square")

    def save_current_frame(self):
        if self.current_frame is not None:
            self.current_frame.save("saved_image.jpg")
            self.alert_label.config(
                text="JPG was saved", fg="green")

    def save_gif(self):
        if len(self.frames_for_gif) >= TOTAL_FRAMES:
            gif_frames = [Image.fromarray(frame) for frame in self.frames_for_gif]
            gif_frames[0].save("saved_gif.gif", save_all=True, append_images=gif_frames[1:], duration=100, loop=0)

    def start_gif_making(self):
        self.gif_state = 1
        self.frames_for_gif = []

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    WIDTH, HEIGHT = 640, 480
    TOTAL_FRAMES = 5
    SAVE_GIF_EVERY = 30
    app = CameraApp(width=WIDTH, height=HEIGHT)
    app.start()
    app.cleanup()
