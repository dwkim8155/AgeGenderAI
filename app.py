import os
import cv2
import glob
import numpy as np
from PIL import Image
from yoloface import face_analysis
from flask import Flask, request, jsonify
from tf_keras.models import load_model

app = Flask(__name__)
root = os.getcwd()
model_age = load_model(os.path.join(root, 'model_age.hdf5'))
model_gender = load_model(os.path.join(root, 'model_gender.hdf5'))

label_gender = {0: 'Male', 1: 'Female'}

def detect_image_with_crop(img_path, save_path, model_age=model_age, model_gender=model_gender):
    img_name = os.path.basename(img_path)
    ori_img = cv2.imread(img_path)
    img = cv2.imread(img_path)
    # face detection box
    face = face_analysis()
    _, boxes, _ = face.face_detection(image_path=img_path, model='full')
    faces = []
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
        img_detect = cv2.resize(ori_img[y:y + w, x:x + h], (50, 50)).reshape(1, 50, 50, 3)
        
        # Detect Gender
        gender_arg = np.round(model_gender.predict(img_detect / 255.)).astype(np.uint8)
        gender = label_gender[gender_arg[0][0]]
        # gender = label_gender[0]
        # Detect Age
        if gender == 'Male':
            age = int(np.round(model_age.predict(img_detect / 255.))[0][0])
            # age = 20
            age = int(age * 1.35) if age < 20 else int(age * 1.25)
        else:
            age = int(np.round(model_age.predict(img_detect / 255.))[0][0])
            if age < 10:
                age = int(age * 4)
            elif age < 20:
                age = int(age * 2)
            else:
                age = int(age * 1.25)
        
        info = f'Age:{age}_{gender}'
        font_scale = w / 280
        # 음영 (흰색 배경)
        cv2.putText(img, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=3)
        
        # 텍스트 (초록색 글자)
        cv2.putText(img, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness=1)
        
        # Crop the face and save
        face_img = ori_img[y:y + w, x:x + h]
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img_rgb)
        face_path = os.path.join(save_path, f'face_{idx + 1}.png')
        face_pil.save(face_path)
        faces.append((age, gender, face_path))
    
    # BGR to RGB for the full image
    saved_path = os.path.join(save_path, f'result_{img_name}')            
    cv2.imwrite(saved_path, img)    
    return faces


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    file_path = data['file_path']
    save_path = data['saved_path']
    
    faces = detect_image_with_crop(file_path, save_path)
    return jsonify({'faces': faces})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
