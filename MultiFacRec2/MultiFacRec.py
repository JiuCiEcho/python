import cv2
import os
import numpy as np

# 获取数据集文件夹中的人脸图片
data_dir = './dataset'
people = []
faces = []
labels = []
for folder_name in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    people.append(folder_name)
    images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        images.append(image)
    faces.append(np.array(images))
    labels.append([len(people)-1]*len(images))
faces = np.concatenate(faces, axis=0)
labels = np.concatenate(labels, axis=0)
np.save('faces.npy', faces)
np.save('labels.npy', labels)

# 训练人脸识别模型
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, labels)

# 进行实时人脸识别
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces_rects:
        face = gray[y:y+h, x:x+w]
        # 对每个检测到的人脸进行多次识别，并取平均值
        predictions = []
        for i in range(5):
            label, confidence = face_recognizer.predict(face)
            if confidence < 100:
                predictions.append(label)
        if len(predictions) > 0:
            label = int(np.mean(predictions))
            name = people[label]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=color, thickness=2)
    cv2.imshow('face recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""
import os
import cv2
import numpy as np

# 加载人脸识别模型
face_model = cv2.face.LBPHFaceRecognizer_create()
face_model.read('face_model.xml')

# 加载人脸检测器和关键点定位器
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_landmark = cv2.face.createFacemarkLBF()
face_landmark.loadModel('lbfmodel.yaml')

# 加载数据集
dataset_path = 'dataset'
names = os.listdir(dataset_path)

# 定义数据增强函数
def data_augmentation(img):
    flip_img = cv2.flip(img, 1)  # 水平翻转
    rotate_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 顺时针旋转90度
    return [img, flip_img, rotate_img]

# 定义人脸对齐函数
def face_alignment(img, landmarks):
    # 获取人脸关键点坐标
    points = []
    for landmark in landmarks:
        points.append((landmark[0][0], landmark[0][1]))
    points = np.array(points)

    # 计算仿射变换矩阵
    desired_left_eye = (0.35, 0.35)
    desired_face_width = 256
    desired_face_height = 256
    M = cv2.estimateAffinePartial2D(points, np.array([(desired_face_width/2, desired_face_height*desired_left_eye[1]),
                                                      (desired_face_width/2, desired_face_height*(1-desired_left_eye[1])),
                                                      (desired_face_width*desired_left_eye[0], desired_face_height/2)]))[0]

    # 对人脸图像进行仿射变换
    aligned_face = cv2.warpAffine(img, M, (desired_face_width, desired_face_height))
    return aligned_face

# 定义特征提取函数
def extract_feature(img):
    # 使用LBP算法提取人脸特征
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = cv2.LBP_create()
    hist = cv2.calcHist([cv2.LBP(gray, 8, 1, lbp)], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 定义模型融合函数
def model_fusion(models, img):
    features = []
    for model in models:
        feature = model.predict(extract_feature(img))
        features.append(feature)
    features = np.array(features)
    return np.argmax(np.sum(features, axis=0))

# 实时人脸识别
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        # 人脸检测和关键点定位
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        _, landmarks = face_landmark.fit(gray, faces)

        # 人脸识别
        for i, face in enumerate(faces):
            x, y, w, h = face
            # 人脸对齐和特征提取
            aligned_face = face_alignment(frame[y:y+h, x:x+w], landmarks[i])
            features = extract_feature(aligned_face)

            # 模型融合
            predictions = model_fusion([face_model], aligned_face)
            name = names[predictions]

            # 绘制人脸边框和姓名
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
"""


