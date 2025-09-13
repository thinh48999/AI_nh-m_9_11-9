import cv2
import numpy as np
from keras.models import load_model
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os

model = load_model("face_mnist_style.keras")
class_names = ["Huy", "Khai", "Thinh"]
person_info = {
    "Huy": {
        "Tên đầy đủ": "Đồng Nguyễn Nhật Huy",
        "Tuổi": "19",
        "Nghề nghiệp": "Sinh viên Đại học",
        "Sở thích": "Lập trình, Chơi game, Đá bóng",
        "Mô tả": "Huy là một sinh viên xuất sắc chuyên ngành Công nghệ Thông tin. Cậu ấy có niềm đam mê với lập trình và thường tham gia các cuộc thi hackathon. Với tính cách thân thiện và khả năng giải quyết vấn đề tốt, Huy luôn được bạn bè và thầy cô yêu quý."
    },
    "Khai": {
        "Tên đầy đủ": "Phan Trần Quang Khải",
        "Tuổi": "19",
        "Nghề nghiệp": "Kỹ sư phần mềm",
        "Sở thích": "Nhiếp ảnh, Du lịch, Âm nhạc",
        "Mô tả": "Khai là một kỹ sư phần mềm tài năng với 2 năm kinh nghiệm tại công ty công nghệ hàng đầu. Anh ấy yêu thích chụp ảnh phong cảnh và đã đi đến nhiều nơi trên đất nước Việt Nam. Khai có khả năng lãnh đạo tốt và luôn sẵn sàng giúp đỡ đồng nghiệp."
    },
    "Thinh": {
        "Tên đầy đủ": "Trần Văn Thịnh",
        "Tuổi": "19",
        "Nghề nghiệp": "Thiết kế đồ họa",
        "Sở thích": "Vẽ tranh, Đọc sách, Nấu ăn",
        "Mô tả": "Thịnh là một designer sáng tạo với con mắt thẩm mỹ tốt. Cậu ấy đã tham gia nhiều dự án thiết kế lớn và luôn tìm tòi những xu hướng mới. Thịnh có khả năng làm việc độc lập và theo nhóm rất tốt, luôn mang đến những giải pháp thiết kế độc đáo và ấn tượng."
    }
}


class FaceRecognitionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Nhận Diện Khuôn Mặt")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Biến lưu trữ
        self.image_path = None
        self.cap = None
        self.webcam_running = False
        self.current_frame = None
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
        title_label = ttk.Label(main_frame, text="HỆ THỐNG NHẬN DIỆN KHUÔN MẶT",
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

        # Image label for displaying selected image or webcam - FIXED SIZE
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
                h, w = frame.shape[:2]
                label_width = self.image_label.winfo_width()
                label_height = self.image_label.winfo_height()
                if label_width <= 1:
                    label_width = 400
                    label_height = 300

                ratio = min(label_width / w, label_height / h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                if new_w < 100:
                    new_w = 100
                if new_h < 100:
                    new_h = 100

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
        if len(image.shape) == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        if label_width <= 1:
            label_width = 400
            label_height = 300

        w, h = img_pil.size
        ratio = min(label_width / w, label_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        if new_w < 100:
            new_w = 100
        if new_h < 100:
            new_h = 100

        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img_resized)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def capture_and_recognize(self):
        if self.webcam_image is not None:
            self.original_image = self.webcam_image.copy()
            self.display_image(self.original_image)
            self.path_label.configure(text=f"Đã chụp ảnh từ webcam")
            self.recognize_face(self.webcam_image, source="webcam")
        else:
            messagebox.showwarning("Cảnh báo", "Không có frame nào từ webcam!")

    def select_and_recognize(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh khuôn mặt",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if self.original_image is None:
                messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
                return
            self.display_image(self.original_image)
            try:
                self.path_label.configure(text=f"Đã chọn: {os.path.basename(file_path)}")
                self.recognize_face(self.original_image, source="file")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể đọc ảnh: {str(e)}")

    def recognize_face(self, image, source="file"):
        if image is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để nhận diện!")
            return

        try:
            img_processed = image.copy()
            if len(img_processed.shape) == 3:
                img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
            img_processed = cv2.resize(img_processed, (60, 60))
            img_processed = cv2.equalizeHist(img_processed)
            img_processed = cv2.GaussianBlur(img_processed, (3, 3), 0)
            img_processed = img_processed.astype("float32") / 255.0
            img_processed = img_processed.reshape(1, 60 * 60)

            predictions = model.predict(img_processed, verbose=0)
            class_idx = np.argmax(predictions)
            predicted_name = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            self.result_var.set(
                f"✅ NHẬN DIỆN THÀNH CÔNG!\n\nKết quả: {predicted_name}\nĐộ chính xác: {confidence:.2f}%")

            self.show_person_info(predicted_name)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")

    def show_person_info(self, person_name):
        info = person_info.get(person_name, {})

        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)

        # Display person information with formatting
        self.info_text.insert(tk.END, f"THÔNG TIN CHI TIẾT - {person_name.upper()}\n\n", "title")
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
    app = FaceRecognitionUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_webcam(), root.destroy()))
    root.mainloop()