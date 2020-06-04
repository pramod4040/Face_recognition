import cv2
import os
import numpy as np

subjects = ["", "Pramod", "Muna", "Pramisha"]


#setting variable

prepare_train_data = False
train_model = False
#make detect_people_in_photo false if you wanna detect in video from webcamp
detect_people_in_photo = True



def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0]



# detect multiple faces
def detect_faces(img):
    detected_faces = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    faces_all = face_cascade.detectMultiScale(gray, 1.2, 5)
    for face in faces_all:
        (x,y,w,h) = face
        detected_f = [gray[y:y+w, x:x+h], face]
        detected_faces.append(detected_f)
    return detected_faces



def prepare_traning_data(data_folder_path):
    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_name = os.listdir(subject_dir_path)

        for image_name in subject_images_name:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            print("Traning On Image...")
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
    return faces, labels


if prepare_train_data:
    print("preparing Data .....")
    faces, labels = prepare_traning_data("Data")
    print("Total Faces:", len(faces))
    print("Total Labels:", len(labels))


print("Data Prepared")


def train_data(faces, labels):
    face_recog = cv2.face.LBPHFaceRecognizer_create()
    face_recog.train(faces, np.array(labels))
    face_recog.save("model_face_recog_ppm.yml")
    return

if prepare_train_data & train_model:
    print('traning data')
    train_data(faces, labels)



print("Data Trained")


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("model_face_recog_ppm.yml")



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    det_faces = detect_faces(img)

    for d_face, rect in det_faces:
        label = face_recognizer.predict(d_face)
        label_text = subjects[label[0]]
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1] - 5)

    # label = face_recognizer.predict(face)
    # label_text = subjects[label[0]]
    #
    # draw_rectangle(img, rect)
    # draw_text(img, label_text, rect[0], rect[1] - 5)
    return img


# test_img1 = cv2.imread("test_data/1.jpg")
# test_img2 = cv2.imread("test_data/2.jp g")

def detect_people_from_image():
    both = cv2.imread("test_data/pp.jpg")
    p3 = predict(both)
    cv2.imshow('Photo', p3)


if detect_people_in_photo:
    detect_people_from_image()


print("Lets Rock and Roll...")

cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img1 = predict(img)

    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # for x, y, w,h in faces:
    # cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0,0 ), 2)
    cv2.imshow('Video Going', img1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()