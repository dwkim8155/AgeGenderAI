import os
import time
import requests
import cv2
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from yoloface import face_analysis

def save_uploaded_file(uploaded_file):
    # Save the uploaded file to a temporary directory
    try:
        os.makedirs('tempDir', exist_ok=True)
        file_path = os.path.join('tempDir', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def call_model_server(file_path, saved_path):
    url = "http://localhost:5001/predict"
    response = requests.post(url, json={'file_path': file_path, 'saved_path': saved_path})
    return response.json()

def main():
    st.title("Age and Gender Prediction")
    
    st.write("나이와 성별을 예측할 사진을 업로드해주세요")

    # Photo upload
    uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        
        # 구분선 추가
        st.markdown("---")
        
        # 제목을 가운데 정렬하고 크게 만들기
        st.markdown("<h2 style='text-align: center;'>성별 및 얼굴 나이 예측 결과</h2>", unsafe_allow_html=True)
        
        with st.spinner('Classifying...'):
            progress_bar = st.progress(0)
            
            saved_path = os.path.join(os.getcwd(),'output')
            img_name = file_path.split('/')[-1]
            
            faces = call_model_server(file_path, saved_path)['faces']
            
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)  # Simulate some work being done
        
        result_img = Image.open(os.path.join(saved_path, f"result_{img_name}"))
        # Display the result
        st.image(result_img, caption='Predicted Result', use_column_width=True)
        
        # Provide download button for the result image
        
        with open(os.path.join(saved_path, f"result_{img_name}"), "rb") as file:
            st.download_button(
                label="Download Result Image",
                data=file,
                file_name=f"result_{img_name}",
                mime="image/png"
            )
            
        # Center align the download button using flexbox
        st.markdown(
            """
            <style>
            div.stDownloadButton { display: flex; justify-content: center; }
            div.st-emotion-cache-1kyxreq { display: flex; justify-content: center; }
            h3 { text-align: center; }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Prepare data for the table
        data = []
        for idx, (age, gender, face_pil) in enumerate(faces):
            face_id = f"ID{idx + 1}"
            data.append({"ID": face_id, "Age": age, "Gender": gender, "Image": face_pil})
        
        # Create a dataframe
        df = pd.DataFrame(data)
        
        # Display the table
        st.markdown("<h3 style='text-align: center;'>예측 결과 표</h3>", unsafe_allow_html=True)
        st.table(df[['ID', 'Age', 'Gender']])
        
        # Display cropped face images with IDs
        for idx, row in df.iterrows():
            st.markdown(f"### {row['ID']}")
            st.image(row['Image'], caption=f"Age: {row['Age']}, Gender: {row['Gender']}", width=300)
        
    
if __name__ == "__main__":
    main()