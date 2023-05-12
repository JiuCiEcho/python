import cv2
import os
import numpy as np

def load_dataset(data_dir):
    """
    从数据集文件夹中获取人脸图片。
    """
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
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            images.append(gray)
        faces.append(np.array(images))
        labels.append([len(people)-1]*len(images))
    faces = np.concatenate(faces, axis=0)
    labels = np.concatenate(labels, axis=0)
    return people, faces, labels

class FaceRecognizer:
    def __init__(self, data_dir):
        """
        初始化人脸识别器，并训练人脸识别模型。
        """
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.people, faces, labels = load_dataset(data_dir)
        self.face_recognizer.train(faces, labels)

    def recognize(self, frame):
        """
        对实时捕获的帧进行人脸识别。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_rects:
            face = gray[y:y+h, x:x+w]
            # 对每个检测到的人脸进行多次识别，并取平均值
            predictions = []
            for i in range(5):
                label, confidence = self.face_recognizer.predict(face)
                if confidence < 100:
                    predictions.append(label)
            if len(predictions) > 0:
                label = int(np.mean(predictions))
                name = self.people[label]
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
            # 在帧上标出人脸和人名
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        return frame

if __name__ == "__main__":
    recognizer = FaceRecognizer("dataset")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = recognizer.recognize(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
