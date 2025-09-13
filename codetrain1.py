import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#xử lý thư mục, file của data
def load_data(data_dir, img_size=(60, 60), test_size=0.2):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for file in os.listdir(cls_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".HEIC")):
                img_path = os.path.join(cls_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # Tiền xử lý ảnh
                img = cv2.resize(img, img_size)
                img = cv2.equalizeHist(img)  # Cân bằng histogram để cải thiện độ tương phản
                img = cv2.GaussianBlur(img, (3, 3), 0)  # Làm mờ để giảm nhiễu
                img = img.astype("float32") / 255.0
                X.append(img)
                y.append(class_map[cls])

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], img_size[0] * img_size[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return (X_train, y_train), (X_test, y_test), class_names


def create_model(input_shape, num_classes):
    model = Sequential()

    # Mạng neural sâu hơn với các kỹ thuật chống overfitting
    model.add(Dense(1024, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    # Sử dụng optimizer tốt hơn
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def augment_data(X_train, y_train):
    # Tạo dữ liệu augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,  # Xoay ảnh ngẫu nhiên ±20 độ
        width_shift_range=0.2,  # Dịch chuyển ngang ngẫu nhiên 20%
        height_shift_range=0.2,  # Dịch chuyển dọc ngẫu nhiên 20%
        shear_range=0.2,  # Biến dạng cắt ngẫu nhiên 20%
        zoom_range=0.2,  # Zoom ngẫu nhiên ±20%
        horizontal_flip=True,  # Lật ngang ngẫu nhiên
        fill_mode='nearest'  # Điền pixel gần nhất khi biến đổi
    )

    # Reshape dữ liệu để phù hợp với ImageDataGenerator
    X_train_reshaped = X_train.reshape(-1, 60, 60, 1)

    # Tạo dữ liệu augmentation
    aug_iter = datagen.flow(X_train_reshaped, y_train, batch_size=32)

    # Tạo batch dữ liệu augmentation
    X_augmented = []
    y_augmented = []

    for i in range(len(X_train) // 32):  # Tạo thêm dữ liệu gấp đôi
        X_batch, y_batch = next(aug_iter)
        X_augmented.append(X_batch.reshape(-1, 60 * 60))
        y_augmented.append(y_batch)

    # Kết hợp dữ liệu gốc và dữ liệu augmentation
    X_combined = np.vstack([X_train] + X_augmented)
    y_combined = np.vstack([y_train] + y_augmented)

    return X_combined, y_combined


def plot_training_history(history):
    # Vẽ đồ thị accuracy và loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# Main training code
try:
    data_dir = 'train'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại!")

    # Tải dữ liệu
    (X_train, y_train), (X_test, y_test), class_names = load_data(data_dir)

    # Chuyển đổi nhãn sang dạng categorical
    y_train_categorical = to_categorical(y_train, num_classes=len(class_names))
    y_test_categorical = to_categorical(y_test, num_classes=len(class_names))

    print(f"Số người nhận dạng: {len(class_names)}")
    print(f"Tên người nhận dạng: {class_names}")
    print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")
    print(f"Kích thước dữ liệu kiểm tra: {X_test.shape}")

    # Tạo và huấn luyện mô hình
    model = create_model((60 * 60,), len(class_names))
    model.summary()

    # Augmentation dữ liệu
    X_train_aug, y_train_aug = augment_data(X_train, y_train_categorical)
    print(f"Kích thước dữ liệu sau augmentation: {X_train_aug.shape}")

    # Huấn luyện mô hình
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_categorical),
        verbose=1,
        callbacks=[
            # Dừng sớm nếu validation loss không cải thiện sau 10 epochs
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            # Giảm learning rate nếu validation loss không cải thiện
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
    )

    #Vẽ đồ thị quá trình huấn luyện
    plot_training_history(history)

    # Đánh giá mô hình
    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")

    #Dự đoán
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)

    model.save("face_mnist_style.keras")    #Lưu mô hình

    #Kiểm tra độ chính xác trên từng lớp (người cần nhận diện)
    class_accuracy = {}
    for i, cls_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_classes[class_mask] == y_true[class_mask])
            class_accuracy[cls_name] = class_acc
            print(f"Độ chính xác khi huấn luyện data của {cls_name}: {class_acc:.4f}")

except Exception as e: #khi train không thành công thì in ra lỗi
    print(f"Lỗi: {e}")
    import traceback
    traceback.print_exc()