import cv2
import os
import numpy as np

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 从文件夹中读取每张图片，对图片上的人脸进行检测标记，并重新保存添加过标记的图片
def detect_and_save_faces(folder_path):
    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(folder_path):
        # 读取图像文件
        img_path = os.path.join(folder_path, filename)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        # 在人脸周围画矩形并保存图像
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "Detected face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(folder_path, "detected_" + filename))
        else:
            cv2.putText(img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imencode('.jpg', img)[1].tofile(os.path.join(folder_path, "undetected_"+filename))

# 测试
if __name__ == '__main__':
    detect_and_save_faces("data")


