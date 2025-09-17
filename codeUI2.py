import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

# Tải mô hình
model = load_model("money_detect_mnist_style.keras")
class_names = ["000100", "000200", "000500", "001000", "002000", "005000",
               "010000", "020000", "050000", "100000", "200000", "500000"]

# Sửa lại cấu trúc person_info
person_info = {
    "000100": {"Mệnh giá": "100 đồng", "Mô tả": "Tiền polymer mệnh giá nhỏ"},
    "000200": {"Mệnh giá": "200 đồng", "Mô tả": "Tiền polymer"},
    "000500": {"Mệnh giá": "500 đồng", "Mô tả": "Tiền polymer"},
    "001000": {"Mệnh giá": "1.000 đồng", "Mô tả": "Tiền polymer"},
    "002000": {"Mệnh giá": "2.000 đồng", "Mô tả": "Tiền polymer"},
    "005000": {"Mệnh giá": "5.000 đồng", "Mô tả": "Tiền polymer"},
    "010000": {"Mệnh giá": "10.000 đồng", "Mô tả": "Tiền polymer, màu nâu đỏ"},
    "020000": {"Mệnh giá": "20.000 đồng", "Mô tả": "Tiền polymer, màu xanh lam"},
    "050000": {"Mệnh giá": "50.000 đồng", "Mô tả": "Tiền polymer, màu đỏ tía"},
    "100000": {"Mệnh giá": "100.000 đồng", "Mô tả": "Tiền polymer, màu xanh lá"},
    "200000": {"Mệnh giá": "200.000 đồng", "Mô tả": "Tiền polymer, màu nâu vàng"},
    "500000": {"Mệnh giá": "500.000 đồng", "Mô tả": "Tiền polymer, màu xanh dương"}
}


class BanknoteRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Nhận Diện Tiền Tệ Việt Nam")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Biến lưu trữ
        self.image_path = None
        self.cap = None
        self.webcam_running = False
        self.photo = None
        self.original_image = None
        self.webcam_image = None
        self.create_widgets()
        self.start_webcam()

    def create_widgets(self):
        # Configure style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Title.TLabel', background='#f0f0f0', font=('Arial', 20, 'bold'), foreground='#2c3e50')
        style.configure('Result.TLabel', background='#f0f0f0', font=('Arial', 18, 'bold'), foreground='#27ae60')
        style.configure('TButton', font=('Arial', 12), padding=8)
        style.configure('TLabelFrame', font=('Arial', 12, 'bold'), background='#f0f0f0')

        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="HỆ THỐNG NHẬN DIỆN TIỀN TỆ VIỆT NAM",
                                style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Left frame for controls and info
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 20))

        # Right frame for image/webcam
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Selection frame
        select_frame = ttk.Frame(left_frame)
        select_frame.grid(row=0, column=0, pady=(0, 20), sticky=(tk.W, tk.E))

        # Select button
        self.select_btn = ttk.Button(select_frame, text="📁 CHỌN ẢNH VÀ NHẬN DIỆN",
                                     command=self.select_and_recognize, width=25)
        self.select_btn.grid(row=0, column=0, padx=(0, 10))

        # Webcam button
        self.webcam_btn = ttk.Button(select_frame, text="📷 BẬT/TẮT WEBCAM",
                                     command=self.toggle_webcam, width=20)
        self.webcam_btn.grid(row=0, column=1)

        # Path label
        self.path_label = ttk.Label(select_frame, text="Chưa chọn ảnh nào",
                                    foreground="#7f8c8d", font=('Arial', 11))
        self.path_label.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)

        # Result frame
        result_frame = ttk.LabelFrame(left_frame, text="KẾT QUẢ NHẬN DIỆN", padding="15")
        result_frame.grid(row=1, column=0, pady=(0, 20), sticky=(tk.W, tk.E))

        # Result display - centered and larger
        self.result_var = tk.StringVar(value="🔄 Chưa có kết quả")
        result_display = tk.Label(result_frame, textvariable=self.result_var,
                                  font=('Arial', 18, 'bold'), foreground="#2c3e50",
                                  background='#ecf0f1', justify=tk.CENTER,
                                  wraplength=400, padx=20, pady=20,
                                  relief=tk.RIDGE, borderwidth=2)
        result_display.grid(row=0, column=0, pady=10)

        # Information frame
        info_frame = ttk.LabelFrame(left_frame, text="THÔNG TIN CHI TIẾT", padding="15")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Text box for information
        self.info_text = tk.Text(info_frame, width=60, height=15, font=('Arial', 11),
                                 wrap=tk.WORD, state=tk.DISABLED, bg='#ffffff',
                                 relief=tk.FLAT, borderwidth=1, padx=15, pady=15)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars for text box
        v_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        h_scrollbar = ttk.Scrollbar(info_frame, orient=tk.HORIZONTAL, command=self.info_text.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.info_text.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Image/Webcam display frame
        display_frame = ttk.LabelFrame(right_frame, text="XEM TRƯỚC ẢNH/WEBCAM", padding="10")
        display_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Image label for displaying selected image or webcam
        self.image_label = tk.Label(display_frame, bg="#2c3e50", width=50, height=30,
                                    relief=tk.SUNKEN, text="Webcam đang khởi động...")
        self.image_label.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid to expand the label
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)

        # Capture button
        self.capture_btn = ttk.Button(display_frame, text="📸 CHỤP ẢNH VÀ NHẬN DIỆN",
                                      command=self.capture_and_recognize, width=30)
        self.capture_btn.grid(row=1, column=0, pady=10)

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(2, weight=1)

        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

    def start_webcam(self):
        """Khởi động webcam"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể truy cập webcam!")
                return

            self.webcam_running = True
            self.update_webcam()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động webcam: {str(e)}")

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        self.cap = None

    def toggle_webcam(self):
        if self.webcam_running:
            self.stop_webcam()
            self.webcam_btn.configure(text="📷 BẬT WEBCAM")
            self.image_label.configure(image='', text="Webcam đã tắt", bg="#2c3e50", fg="white")
        else:
            self.start_webcam()
            self.webcam_btn.configure(text="📷 TẮT WEBCAM")

    def update_webcam(self):
        if self.webcam_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.webcam_image = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Get current label size or use default
                label_width = self.image_label.winfo_width()
                label_height = self.image_label.winfo_height()
                if label_width <= 1:
                    label_width = 400
                if label_height <= 1:
                    label_height = 300

                # Resize maintaining aspect ratio
                h, w = frame.shape[:2]
                ratio = min(label_width / w, label_height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)

                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                self.image_label.configure(image=self.photo, text="")

            if self.webcam_running:
                self.root.after(10, self.update_webcam)
        else:
            self.image_label.configure(image='', text="Không có tín hiệu webcam", bg="#2c3e50", fg="white")

    def display_image(self, image):
        if image is None:
            return

        # Convert to RGB if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # Already RGB or other
                img_rgb = image
        else:  # Grayscale
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(img_rgb)

        # Get label dimensions
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        if label_width <= 1:
            label_width = 400
        if label_height <= 1:
            label_height = 300

        # Resize maintaining aspect ratio
        w, h = img_pil.size
        ratio = min(label_width / w, label_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def capture_and_recognize(self):
        if self.webcam_image is not None:
            self.original_image = self.webcam_image.copy()
            self.display_image(self.original_image)
            self.path_label.configure(text="Đã chụp ảnh từ webcam")
            self.recognize_banknote(self.webcam_image)
        else:
            messagebox.showwarning("Cảnh báo", "Không có frame nào từ webcam!")

    def select_and_recognize(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh tiền",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                # Try alternative reading method for special characters
                self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.original_image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
            self.display_image(self.original_image)
            try:
                self.path_label.configure(text=f"Đã chọn: {os.path.basename(file_path)}")
                self.recognize_banknote(self.original_image)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc ảnh: {str(e)}")

    def preprocess_image_for_model(self, image):
        """Preprocess image to match model's expected input format"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to 60x60 (model training size)
        resized = cv2.resize(gray, (60, 60))

        # Apply the same preprocessing as during training
        resized = cv2.equalizeHist(resized)  # Histogram equalization
        resized = cv2.GaussianBlur(resized, (3, 3), 0)  # Gaussian blur

        # Normalize
        normalized = resized.astype("float32") / 255.0

        # Flatten to 1D vector (3600 elements)
        flattened = normalized.flatten()

        # Reshape to match model input shape (batch_size, 3600)
        return flattened.reshape(1, -1)

    def recognize_banknote(self, image):
        if image is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để nhận diện!")
            return

        try:
            # Preprocess image for the model
            img_processed = self.preprocess_image_for_model(image)

            # Make prediction
            predictions = model.predict(img_processed, verbose=0)
            class_idx = np.argmax(predictions)
            predicted_name = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            self.result_var.set(
                f"✅ NHẬN DIỆN THÀNH CÔNG!\n\nMệnh giá: {predicted_name}\nĐộ chính xác: {confidence:.2f}%")

            self.show_banknote_info(predicted_name)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra khi nhận diện: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_banknote_info(self, banknote_name):
        info = person_info.get(banknote_name, {})

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        # Display banknote information with formatting
        self.info_text.insert(tk.END, f"THÔNG TIN CHI TIẾT - {banknote_name}\n\n", "title")
        self.info_text.tag_configure("title", font=('Arial', 14, 'bold'), foreground='#2980b9')

        for key, value in info.items():
            self.info_text.insert(tk.END, f"• {key}: ", "bold")
            self.info_text.tag_configure("bold", font=('Arial', 11, 'bold'))
            self.info_text.insert(tk.END, f"{value}\n\n")

        self.info_text.configure(state=tk.DISABLED)

    def __del__(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = BanknoteRecognitionUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_webcam(), root.destroy()))
    root.mainloop()