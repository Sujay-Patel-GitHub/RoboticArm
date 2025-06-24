import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1).astype("float32") / 255
x_test = np.expand_dims(x_test, -1).astype("float32") / 255

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 2. Define a deeper CNN with dropout and batchnorm
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.summary()

# 3. Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# 4. Train model with augmentation
batch_size = 128
epochs = 15
model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
    verbose=2
)

# 5. Evaluate performance
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# 6. Webcam inference
cap = cv2.VideoCapture(1)
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    morph = cv2.morphologyEx(
        cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel),
        cv2.MORPH_OPEN, kernel
    )
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area < 200 or area > 20000: continue
        ar = w / float(h)
        if ar < 0.2 or ar > 1.2: continue

        roi = morph[y:y+h, x:x+w]
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = roi
        img = cv2.resize(square, (28, 28)).astype("float32") / 255
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img, verbose=0)[0]
        cls = np.argmax(pred)
        conf = pred[cls]

        if conf < 0.90: continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls} {conf*100:.0f}%",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("CNN Augmented", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
