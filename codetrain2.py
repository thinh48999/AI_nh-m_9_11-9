import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_data(data_dir, img_size=(60, 60), test_size=0.2):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_map = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        for file in os.listdir(cls_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".HEIC")):
                img_path = os.path.join(cls_path, file)
                try:
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    print(f"Error decoding image {img_path}: {e}")
                    continue

                img = cv2.resize(img, img_size)
                img = cv2.equalizeHist(img)
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img = img.astype("float32") / 255.0
                X.append(img)
                y.append(class_map[cls])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], img_size[0] * img_size[1]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    return (X_train, y_train), (X_test, y_test), class_names


def create_model(input_shape, num_classes, img_size=(60, 60)):
    model = Sequential([
        tf.keras.layers.Reshape((img_size[0], img_size[1], 1), input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def augment_data(X_train, y_train, img_size=(60, 60)):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    X_train_reshaped = X_train.reshape(-1, img_size[0], img_size[1], 1)
    aug_iter = datagen.flow(X_train_reshaped, y_train, batch_size=32)

    X_augmented = []
    y_augmented = []

    for i in range(len(X_train) // 32):
        X_batch, y_batch = next(aug_iter)
        X_augmented.append(X_batch.reshape(-1, img_size[0] * img_size[1]))
        y_augmented.append(y_batch)

    X_combined = np.vstack([X_train] + X_augmented)
    y_combined = np.concatenate([y_train] + y_augmented)

    return X_combined, y_combined


def plot_training_history(history):
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
    plt.show()


try:
    data_dir = "train"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục {data_dir} không tồn tại!")

    (X_train, y_train), (X_test, y_test), class_names = load_data(data_dir)
    img_size = (60, 60)

    y_train_categorical = to_categorical(y_train, num_classes=len(class_names))
    y_test_categorical = to_categorical(y_test, num_classes=len(class_names))

    print(f"Số người nhận dạng: {len(class_names)}")
    print(f"Tên người nhận dạng: {class_names}")
    print(f"Kích thước dữ liệu huấn luyện: {X_train.shape}")
    print(f"Kích thước dữ liệu kiểm tra: {X_test.shape}")

    model = create_model((img_size[0] * img_size[1],), len(class_names), img_size)
    model.summary()

    X_train_aug, y_train_aug = augment_data(X_train, y_train_categorical, img_size)
    print(f"Kích thước dữ liệu sau augmentation: {X_train_aug.shape}")

    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test_categorical),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
    )

    plot_training_history(history)

    test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_categorical, axis=1)

    model.save("money_detect_mnist_style.keras")

    class_accuracy = {}
    for i, cls_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred_classes[class_mask] == y_true[class_mask])
            class_accuracy[cls_name] = class_acc
            print(f"Độ chính xác khi huấn luyện data của {cls_name}: {class_acc:.4f}")

except Exception as e:
    print(f"Lỗi: {e}")
    import traceback

    traceback.print_exc()