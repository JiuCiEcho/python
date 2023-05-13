import os
import cv2
import numpy as np
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

# 数据集路径
data_dir = 'dataset'
face_names = []
data = []
labels = []

# 加载数据集
for subdir in os.listdir(data_dir):
    subpath = os.path.join(data_dir, subdir)
    if os.path.isdir(subpath):
        face_names.append(subdir)
        for filename in os.listdir(subpath):
            filepath = os.path.join(subpath, filename)
            # 读取图片时指定字符集为gbk
            img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (128, 128))
            data.append(img)
            labels.append(face_names.index(subdir))

# 转换为numpy数组
data = np.array(data)
labels = np.array(labels)

# 将标签转换为one-hot编码
num_classes = len(face_names)
labels = np.eye(num_classes)[labels]

# 将数据集分为训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# 构建模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data.reshape(-1, 128, 128, 1), train_labels, epochs=20, batch_size=32, validation_data=(test_data.reshape(-1, 128, 128, 1), test_labels))

# 保存模型
model.save('face-recognition.h5')

# 加载模型
model = models.load_model('face-recognition.h5')

# 进行人脸识别
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (128, 128))
        face_image = face_image.reshape(-1, 128, 128, 1)
        prediction = model.predict(face_image)
        name = face_names[np.argmax(prediction)]
        if np.max(prediction) > 0.9:
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 2))
        else:
            cv2.putText(frame, 'Unknown', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
