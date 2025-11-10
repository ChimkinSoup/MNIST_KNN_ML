import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageTk
import numpy as np

from Main import knn_singular 
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("digit recognition")
        self.root.configure(bg="#f5f5f5")
        
        width = 1300
        height = 700
        self.root.geometry(f"{width}x{height}")
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.canvas_width = 280
        self.canvas_height = 280
        self.bg_color = "#1a1a1a"
        self.paint_color = "#ffffff"
        self.brush_size = 7
        self.popup_open = False  
        self.error_popup = None

        # Main container with padding
        main_frame = tk.Frame(root, bg="#f5f5f5")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)

        title_label = tk.Label(main_frame, text="draw a number", 
                               font=("Segoe UI", 24, "bold"), 
                               bg="#f5f5f5", fg="#2c3e50")
        title_label.pack(pady=(0, 20))

        content_frame = tk.Frame(main_frame, bg="#f5f5f5")
        content_frame.pack()

        canvas_frame = tk.Frame(content_frame, bg="#ffffff", relief=tk.FLAT)
        canvas_frame.pack(side=tk.LEFT, padx=20)
        
        shadow_frame = tk.Frame(canvas_frame, bg="#e0e0e0", width=self.canvas_width+6, height=self.canvas_height+6)
        shadow_frame.pack()
        shadow_frame.pack_propagate(False)
        
        inner_frame = tk.Frame(shadow_frame, bg="#ffffff", width=self.canvas_width+4, height=self.canvas_height+4)
        inner_frame.pack(padx=1, pady=1)
        inner_frame.pack_propagate(False)
        
        self.canvas = tk.Canvas(inner_frame, width=self.canvas_width, height=self.canvas_height, 
                               bg=self.bg_color, highlightthickness=0, relief=tk.FLAT)
        self.canvas.pack(padx=2, pady=2)
        
        canvas_label = tk.Label(canvas_frame, text="canvas\n(where you draw)", 
                               font=("Segoe UI", 10), 
                               bg="#ffffff", fg="#7f8c8d")
        canvas_label.pack(pady=(5, 0))

        preview_frame = tk.Frame(content_frame, bg="#ffffff", relief=tk.FLAT)
        preview_frame.pack(side=tk.LEFT, padx=20)
        
        self.preview_size = 140
        preview_shadow = tk.Frame(preview_frame, bg="#e0e0e0", width=self.preview_size+6, height=self.preview_size+6)
        preview_shadow.pack()
        preview_shadow.pack_propagate(False)
        
        preview_inner = tk.Frame(preview_shadow, bg="#ffffff", width=self.preview_size+4, height=self.preview_size+4)
        preview_inner.pack(padx=1, pady=1)
        preview_inner.pack_propagate(False)
        
        self.preview_canvas = tk.Canvas(preview_inner, width=self.preview_size, height=self.preview_size, 
                                       bg="#ffffff", highlightthickness=0, relief=tk.FLAT)
        self.preview_canvas.pack(padx=2, pady=2)
        
        preview_label = tk.Label(preview_frame, text="preview\n(what the model sees)", 
                                font=("Segoe UI", 10), 
                                bg="#ffffff", fg="#7f8c8d")
        preview_label.pack(pady=(5, 0))

        button_frame = tk.Frame(main_frame, bg="#f5f5f5")
        button_frame.pack(pady=30)

        self.predict_button = tk.Button(button_frame, text="predict [enter]", 
                                        command=self.predict,
                                        font=("Segoe UI", 11, "bold"),
                                        bg="#3498db", fg="#ffffff",
                                        activebackground="#2980b9", activeforeground="#ffffff",
                                        relief=tk.FLAT, padx=30, pady=12,
                                        cursor="hand2", bd=0,
                                        highlightthickness=0,
                                        takefocus=1)
        self.predict_button.pack(side=tk.LEFT, padx=10)

        self.clear_button = tk.Button(button_frame, text="clear", 
                                     command=self.clear,
                                     font=("Segoe UI", 11, "bold"),
                                     bg="#95a5a6", fg="#ffffff",
                                     activebackground="#7f8c8d", activeforeground="#ffffff",
                                     relief=tk.FLAT, padx=30, pady=12,
                                     cursor="hand2", bd=0,
                                     highlightthickness=0,
                                     takefocus=1)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)  # Also draw on click
        
        self.root.bind("<Return>", self.on_enter_key)
        self.root.focus_set()  # Make sure root can receive keyboard events
    
    def close_popup_and_clear(self):
        if self.popup_open and hasattr(self, 'current_popup'):
            self.popup_open = False
            try:
                self.current_popup.destroy()
            except:
                pass
            self.root.focus_set()
            self.root.update_idletasks()
            self.root.deiconify()  # Ensure window is visible
            self.predict_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
        self.clear()
    
    def on_enter_key(self, event):
        if self.popup_open:
            self.close_popup_and_clear()
        else:
            self.predict()

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        # Ensure coordinates are within canvas bounds
        x1 = max(0, min(x1, self.canvas_width))
        y1 = max(0, min(y1, self.canvas_height))
        x2 = max(0, min(x2, self.canvas_width))
        y2 = max(0, min(y2, self.canvas_height))
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)
        self.draw.ellipse([x1, y1, x2, y2], fill=255)
        self.update_preview()

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.preview_canvas.delete("all")

    def preprocess_image(self):
        # Downscale to 28x28
        img_small = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_small = img_small.filter(ImageFilter.SMOOTH)
        # MNIST images are white digits (255) on black background (0)
        # Convert to numpy array of integers
        img_array = np.array(img_small, dtype=np.float32)
        # Flatten to 1D array
        img_array = img_array.flatten()
        return img_array, img_small

    def update_preview(self):
        # For preview, keep the original black background, white drawing
        preview_img = self.image.resize((28, 28), Image.Resampling.NEAREST)
        preview_img = preview_img.resize((self.preview_size, self.preview_size), Image.Resampling.NEAREST)
        
        self.preview_tk = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_tk)

    def predict(self):
        if np.sum(np.array(self.image)) == 0:
            # Single error popup policy
            try:
                if self.error_popup is not None and self.error_popup.winfo_exists():
                    self.error_popup.destroy()
            except:
                pass
            self.error_popup = tk.Toplevel(self.root)
            self.error_popup.title("Error")
            self.error_popup.geometry("300x120")
            self.error_popup.resizable(False, False)
            tk.Label(self.error_popup, text="Please draw a digit first!", 
                    font=("Segoe UI", 11), fg="#e74c3c").pack(pady=20)
            tk.Button(self.error_popup, text="OK",
                     command=lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)),
                     bg="#3498db", fg="#ffffff", relief=tk.FLAT, padx=20, pady=5).pack()
            self.error_popup.protocol("WM_DELETE_WINDOW", lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)))

            return
        
        try:
            img_array, _ = self.preprocess_image()
            prediction = knn_singular(img_array)
            self.popup_open = True
            self.show_prediction_popup(prediction)
        except FileNotFoundError as e:
            # Single error popup policy
            try:
                if self.error_popup is not None and self.error_popup.winfo_exists():
                    self.error_popup.destroy()
            except:
                pass
            self.error_popup = tk.Toplevel(self.root)
            self.error_popup.title("Error")
            self.error_popup.geometry("420x140")
            self.error_popup.resizable(False, False)
            tk.Label(self.error_popup, text="MNIST data files not found!\nPlease ensure data files are in the 'data' directory.", 
                    font=("Segoe UI", 10), fg="#e74c3c", justify=tk.LEFT).pack(pady=20)
            tk.Button(self.error_popup, text="OK",
                     command=lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)),
                     bg="#3498db", fg="#ffffff", relief=tk.FLAT, padx=20, pady=5).pack()
            self.error_popup.protocol("WM_DELETE_WINDOW", lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)))
        except Exception as e:
            # Single error popup policy
            try:
                if self.error_popup is not None and self.error_popup.winfo_exists():
                    self.error_popup.destroy()
            except:
                pass
            self.error_popup = tk.Toplevel(self.root)
            self.error_popup.title("Error")
            self.error_popup.geometry("380x140")
            self.error_popup.resizable(False, False)
            tk.Label(self.error_popup, text=f"An error occurred:\n{str(e)}", 
                    font=("Segoe UI", 10), fg="#e74c3c", justify=tk.LEFT).pack(pady=20)
            tk.Button(self.error_popup, text="OK",
                     command=lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)),
                     bg="#3498db", fg="#ffffff", relief=tk.FLAT, padx=20, pady=5).pack()
            self.error_popup.protocol("WM_DELETE_WINDOW", lambda: (self.error_popup.destroy(), setattr(self, "error_popup", None)))
    
    def show_prediction_popup(self, prediction):
        """Create a minimalistic, animated popup with rounded edges"""
        self.popup_open = True
        self.current_popup = tk.Toplevel(self.root)
        popup = self.current_popup
        popup.title("prediction")
        popup.resizable(False, False)
        
        popup.state("zoomed")
        # popup.attributes("-fullscreen", True)
        
        popup_width = popup.winfo_screenwidth()
        popup_height = popup.winfo_screenheight()
        popup.geometry(f"{popup_width}x{popup_height}+0+0")
        
        canvas = tk.Canvas(popup, width=popup.winfo_screenwidth(), height=popup.winfo_screenheight(), 
                          highlightthickness=0, bg="#f0f0f0")
        canvas.pack()
        
        bg_color = "#ffffff"
        border_color = "#e0e0e0"
        shadow_color = "#d0d0d0"
        
        # Shadow layer (offset by 4 pixels)
        canvas.create_rectangle(4, 4, popup_width, popup_height, 
                               fill=shadow_color, outline="", width=0)
        
        # Main background
        canvas.create_rectangle(0, 0, popup_width-4, popup_height-4, 
                               fill=bg_color, outline=border_color, width=1)
        
        title_font = ("Segoe UI", 12, "normal")
        canvas.create_text((popup_width - 4) // 2, 35, 
                          text="prediction", 
                          font=title_font, 
                          fill="#666666")
        
        prediction_text_ref = [None]
        prediction_font = ("Segoe UI", 72, "bold")
        prediction_text_ref[0] = canvas.create_text(
            (popup_width - 4) // 2, 125, 
            text=str(prediction), 
            font=prediction_font, 
            fill="#2c3e50"
        )
        
        button_width = 100
        button_height = 35
        button_x = ((popup_width - 4) - button_width) // 2
        # Add extra bottom padding to keep button off the bottom edge
        button_y = (popup_height - 4) - 150
        
        button_bg = canvas.create_rectangle(
            button_x, button_y, 
            button_x + button_width, button_y + button_height ,
            fill="#3498db", outline="", width=0
        )
        
        button_text = canvas.create_text(
            (popup_width - 4) // 2, button_y + button_height // 2,
            text="close [enter]", 
            font=("Segoe UI", 11, "normal"),
            fill="#ffffff"
        )
        
        # Button hover effect
        def on_mouse_move(e):
            if (button_x <= e.x <= button_x + button_width and 
                button_y <= e.y <= button_y + button_height):
                canvas.itemconfig(button_bg, fill="#2980b9")
            else:
                canvas.itemconfig(button_bg, fill="#3498db")
        
        def close_popup():
            self.popup_open = False
            popup.destroy()
            # Clear canvas and return focus to main window
            self.clear()
            self.root.focus_set()
            self.root.update_idletasks()
            self.root.deiconify()  # Ensure window is visible
            # Make sure buttons are visible
            self.predict_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL)
        
        def on_button_click(e):
            if (button_x <= e.x <= button_x + button_width and 
                button_y <= e.y <= button_y + button_height):
                close_popup()
        
        canvas.bind("<Motion>", on_mouse_move)
        canvas.bind("<Button-1>", on_button_click)
        
        # Make popup receive keyboard events
        popup.focus_set()
        canvas.focus_set()
        
        # Fade-in animation
        def fade_in(alpha=0.0):
            if alpha < 1.0:
                try:
                    popup.attributes('-alpha', alpha)
                    popup.after(10, lambda: fade_in(alpha + 0.05))
                except:
                    pass
            else:
                popup.attributes('-alpha', 1.0)
        
        # Number animation (scale up effect)
        def animate_number(scale=0.5):
            if scale < 1.0:
                try:
                    # Remove old text
                    if prediction_text_ref[0]:
                        canvas.delete(prediction_text_ref[0])
                    # Create new text with scaling effect
                    prediction_font_scaled = ("Segoe UI", int(72 * scale), "bold")
                    prediction_text_ref[0] = canvas.create_text(
                        (popup_width - 4) // 2, 125, 
                        text=str(prediction), 
                        font=prediction_font_scaled, 
                        fill="#2c3e50"
                    )
                    popup.after(15, lambda: animate_number(scale + 0.05))
                except:
                    pass
        
        # Start animations
        popup.attributes('-alpha', 0.0)
        fade_in()
        animate_number()
        
        # Close on Escape or Enter key
        def close_on_key(e):
            if e.keysym == 'Escape' or e.keysym == 'Return':
                if e.keysym == 'Return':
                    close_popup()
                else:
                    self.popup_open = False
                    popup.destroy()
                    self.root.focus_set()
                    self.root.update_idletasks()
        
        popup.bind('<KeyPress>', close_on_key)
        canvas.bind('<KeyPress>', close_on_key)
        
        def on_popup_destroy():
            self.popup_open = False
        
        popup.protocol("WM_DELETE_WINDOW", lambda: (close_popup(), None))


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = PaintApp(root)
    root.mainloop()
