
import cv2
import os
import numpy as np
class FaceRecognizer:
    def __init__(self, data_dir):
        """
        初始化人脸识别器，并训练人脸识别模型。
        """
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.people = []
        self.faces = []
        self.labels = []
        self.train(data_dir)
    def train(self, data_dir):
        """
        训练人脸识别模型。
        """
        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            self.people.append(folder_name)
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces_rects = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces_rects) == 1:
                    (x, y, w, h) = faces_rects[0]
                    face = gray[y:y + h, x:x + w]
                    face = cv2.resize(face, (100, 100))
                    self.faces.append(face)
                    self.labels.append(len(self.people) - 1)
                else:
                    print("Warning: Skipping image '{}' due to no or multiple faces detected.".format(image_path))
        self.face_recognizer.train(self.faces, np.array(self.labels))
    def recognize(self, frame):
        """
        对实时捕获的帧进行人脸识别。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces_rects:
            face = gray[y:y + h, x:x + w]
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        return frame
if __name__ == "__main__":
    data_dir = "dataset"
    recognizer = FaceRecognizer(data_dir)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = recognizer.recognize(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
