import os

import cv2
import numpy as np
import requests
import torch
from compreface import CompreFace
from compreface.collections import FaceCollection
from compreface.service import RecognitionService
from vidgear.gears import NetGear
from environs import Env

from sface.sface import SFace
from yunet.yunet import YuNet


# все секретные настройки читаются из файла .env, делать по подобию .env_example
env = Env()
env.read_env()

DOMAIN: str = 'http://localhost'
PORT: str = '8000'
RECOGNITION_API_KEY: str = ''

SKIP_FRAMES = 20

AUTO_DETECTION_SIZE = True
FOR_DETECTION_RESIZE = (320, 320)

compre_face: CompreFace = CompreFace(DOMAIN, PORT, {
    "limit": 0,
    "det_prob_threshold": 0.8,
    "prediction_count": 1,
    "status": "true"
})
recognition: RecognitionService = compre_face.init_face_recognition(RECOGNITION_API_KEY)
face_collection: FaceCollection = recognition.get_face_collection()

detector = YuNet(modelPath='./yunet/face_detection_yunet_2022mar.onnx',
                 inputSize=FOR_DETECTION_RESIZE,
                 confThreshold=0.95,
                 nmsThreshold=0.3,
                 topK=5000,
                 backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                 targetId=cv2.dnn.DNN_TARGET_CPU)

recognizer = SFace(modelPath='./sface/face_recognition_sface_2021dec.onnx', disType=0, backendId=cv2.dnn.DNN_BACKEND_OPENCV, targetId=cv2.dnn.DNN_TARGET_CPU)

objects_detection = torch.hub.load('ultralytics/yolov5', 'yolov5x')
BAD_OBJ = {'tv', 'laptop', 'cell phone', 'book'}


def sface_recognize(img1, img2):
    detector.setInputSize([img1.shape[1], img1.shape[0]])
    face1 = detector.infer(img1)
    if face1 is None:
        return 0
    detector.setInputSize([img2.shape[1], img2.shape[0]])
    face2 = detector.infer(img2)
    if face2 is None:
        return 0
    return recognizer.match(img1, face1[0][:-1], img2, face2[0][:-1])


def recognize_face(image_path):
    response = recognition.recognize(image_path)
    result = response.get('result')
    if result and len(result) == 1:
        subjects = result[0]['subjects']
        if subjects and len(subjects) == 1:
            email = subjects[0]['subject']
            similarity = subjects[0]['similarity']
            print('see', email, similarity)
            if similarity > 0.955:
                # неэффективно, заменить
                for face_obj in face_collection.list()['faces']:
                    if face_obj['subject'] == email:
                        image_id = face_obj['image_id']
                        res = requests.get(f'{DOMAIN}:{PORT}/api/v1/recognition/faces/{image_id}/img', headers={'x-api-key': RECOGNITION_API_KEY})
                        if res.status_code == 200:
                            with open('face_that_match.jpg', 'wb') as img1:
                                img1.write(res.content)
                            r = sface_recognize(cv2.imread('face_that_match.jpg'), cv2.imread('face.jpg'))
                            # if not r:
                            #     print('not all match')
                            #     return
                            if r:
                                break
                        else:
                            print('error')
                            return

                r2 = objects_detection('face.jpg')
                if set(r2.pandas().xyxy[0].name) & BAD_OBJ:
                    print('face auth')
                else:
                    print('accept', email, similarity)
                    # r = requests.post("http://127.0.0.1:8001/face_recognize/", json={'secret_key': '12524', 'email': email})
                    # if r.status_code == 500:
                    #     print(email, 'NOT EXIST !!!!!!!!!!')
                    #     exit(0)


client = NetGear(
    address="127.0.0.1",
    port="5454",
    pattern=2,
    receive_mode=True,
    logging=True,
)

img = client.recv()
if AUTO_DETECTION_SIZE:
    img_h, img_w = img.shape[0], img.shape[1]
    detector.setInputSize([img_w, img_h])
else:
    img_h, img_w = FOR_DETECTION_RESIZE[0], FOR_DETECTION_RESIZE[1]

frames_count = 0
while True:
    img = client.recv()

    if frames_count == SKIP_FRAMES:
        frames_count = 0
        if not AUTO_DETECTION_SIZE:
            img = cv2.resize(img, dsize=FOR_DETECTION_RESIZE)

        img_to_show = img.copy()
        results = detector.infer(img)

        if results is not None and len(results) == 1:
            dirname, _ = os.path.split(os.path.abspath(__file__))
            file_path = os.path.join(dirname, 'face.jpg')
            cv2.imwrite(file_path, img)
            recognize_face(file_path)

        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            cv2.rectangle(img_to_show, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        # cv2.imshow('facial detecction in a video', img_to_show)
    else:
        frames_count += 1

    k = cv2.waitKey(30) & 0xff  # wait Esc key
    if k == 27:
        break

cv2.destroyAllWindows()
client.close()
