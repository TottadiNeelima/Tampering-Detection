import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import time
import requests
import cv2
# PRE-PROCESSING THE IMAGE FOR TESTING
from PIL import Image, ImageChops, ImageEnhance
import os
import itertools
import tempfile
from sklearn.cluster import DBSCAN

# st.snow()
st.set_page_config(page_title='Image_Tampering_Detection_And_Localization')
st.markdown( """<style>

MainMenu {visibility: hidden;}
header {visibility:hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"]{
#background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fbackground-image&psig=AOvVaw1eliDlwVXLjqZLXYNn4iLH&ust=1697696999583000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCLCi0Kf8_oEDFQAAAAAdAAAAABAE");
background-color:cover;
}
</style> """, unsafe_allow_html=True)


#st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
#page_bg_img=
#"""
#[data-testid="stAppViewContainer"]{
 #background-image: url("C:\Users\vijja\PycharmProjects\pythonProject43\th (4).jpeg");
  #background-color:cover;
 #}
 #</style>
 #"""
# st.markdown(page_bg_img,unsafe_allow_html=True)
model = load_model("cnn_model.h5")
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file.jpg'
    ela_filename = 'temp_ela_file.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

image_size = (128,128)

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 85).resize(image_size)).flatten() / 255.0
res1=0
def predict(imge):
    ela_img=prepare_image(imge)
    ela_img = ela_img.reshape(-1, 128, 128, 3)
    arr = model.predict(ela_img)
    print(arr)
    if (arr[0][0] > arr[0][1]):
        res1=1
        return "IMAGE IS TAMPERED"
    else:
        return "IMAGE IS AUTHENTICATED";
#Localization code part
def locateForgery(eps, min_sample, descriptors, key_points, image):
    l1=[]
    l2=[]
    clusters = DBSCAN(eps=eps, min_samples=min_sample).fit(descriptors)
    size = np.unique(clusters.labels_).shape[0] - 1
    forgery = image.copy()
    m=0
    if (size == 0) and (np.unique(clusters.labels_)[0] == -1):
        print('No Forgery Found!!')
        return None
    if size == 0:
        size = 1
    cluster_list = [[] for i in range(size)]
    for idx in range(len(key_points)):
        if clusters.labels_[idx] != -1:
            cluster_list[clusters.labels_[idx]].append((int(key_points[idx].pt[0]), int(key_points[idx].pt[1])))
            if m<size:
              l1.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
              m=m+1
              #print(m)
            else:
              l2.append((int(key_points[idx].pt[0]),int(key_points[idx].pt[1])))
    points=l1
    min_x1, min_y1 = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
    max_x1, max_y1 = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]
    #print(min_x1,min_y1,max_x1,max_y1)
    #print(l1)
    #print(l2)
    for i in range((2)):
      # print(l1[i])
      if i==0:
        center_coordinates = min_x1,min_y1
      elif(i==1):
        center_coordinates=max_x1,max_y1
      elif(i==2):
        center_coordinates=max_x1
      elif(i==3):
        center_coordinates=max_y1

    points=l2
    min_x2, min_y2 = min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1]
    max_x2, max_y2 = max(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[1])[1]
    for i in range((2)):
      # print(l1[i])
      if i==0:
        center_coordinates = min_x2,min_y2
      elif(i==1):
        center_coordinates=max_x2,max_y2
      elif(i==2):
        center_coordinates=max_x2
      elif(i==3):
        center_coordinates=max_y2
# Radius of circle
    radius = 3
# Red color in BGR
    color = (0, 0, 255)
# Line thickness of -1 px
    thickness = -1
    # forgery = cv2.circle(forgery, center_coordinates, radius, color, thickness)
    cv2.rectangle(forgery, (min_x1-10, min_y1-10), (max_x1+10, max_y1+10), (0, 0, 255), 2)
    cv2.rectangle(forgery, (min_x2-10, min_y1-10), (max_x2+10, max_y2+10), (0, 0, 255), 2)
    return forgery

def siftDetector(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points, descriptors

def local(uploaded_image):
    # Convert the uploaded image to a PIL Image
    image = Image.open(uploaded_image)

    # Save the PIL Image to a temporary file in JPEG format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, "JPEG")
        temp_file_path = temp_file.name

    # Read the image using OpenCV
    image_cv = cv2.imread(temp_file_path)

    key_points, descriptors = siftDetector(image_cv)
    forgery = locateForgery(40, 2, descriptors, key_points, image_cv)

    # Convert the forgery result back to a PIL Image to display
    forgery_pil = Image.fromarray(cv2.cvtColor(forgery, cv2.COLOR_BGR2RGB))

    return forgery_pil


#def localization_copy_move(imge):
 #   if st.button('Detect Forgery'):
  #          # Call your 'local' function with the uploaded image
   #         forgery_result = local(imge)
    #        if forgery_result:
     #           st.image(forgery_result, caption='Forgery Detected', use_column_width=True)
      #      else:
       #         st.text("No Forgery Detected")


# img_url = st.text_input(label='Enter Image URL')

# if (img_url != "") or (img_url != None):
#     img = Image.open(requests.get(img_url, stream=True).raw)
#     img.save('vvv.jpg')
#     st.image(img)
#Main code part 

st.title('Forgery Detection And Localization on Copy-Move images')
imge=st.file_uploader('Upload your file', type=['JPG', 'PNG', 'JPEG', 'TIF'], accept_multiple_files=False, key=None, help=None,
                 on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
#image_path= imge.name
custom_style = """
<style>
    .big-font {
        font-size: 30px; /* Adjust the font size as needed */
    }
</style>
"""

# Display the prediction with the custom CSS style
st.markdown(custom_style, unsafe_allow_html=True)

if(imge!=None):
    st.image(imge, caption='Uploaded Image')
# if(imge==None and img_url!=None):
#     imeg=img
if st.button('Predict'):
    with st.spinner(text="Checking....."):
        time.sleep(5)
        predict=predict(imge)
        # if(img_url!=None):
        #     predict = predict('vvv.jpg')
    result=st.write(f'<div class="big-font">{predict}</div>', unsafe_allow_html=True)
    #result=st.write(predict,unsafe_allow_html=True)
res="IMAGE IS TAMPERED"
st.write("Please Click the button \" Detect Localization\" when the output of the given input is \"IMAGE IS TAMPERED\" and also when the image is \"copy-move image type\" ")
#if res=="IMAGE IS TAMPERED":
 #   with st.spinner(text="Checking....."):
  #      time.sleep(5)
   #     result=localization_copy_move(imge)
    #    if result is None:
     #       st.write("No valid image to display.")
      #  else:
       #     st.image(result)
def main():
    #uploaded_image =imge
    if imge is not None:
        st.image(imge, caption='Uploaded Image', use_column_width=True)
        if st.button('Detect Localization'):
            # Call your 'local' function with the uploaded image
            forgery_result = local(imge)
            if forgery_result:
                st.image(forgery_result, caption='Forgery Detected', use_column_width=True)
            else:
                st.text("No Forgery Detected")
    else:
        st.text("Please upload an image.")
    
 
if __name__ == '__main__':
    main()   
#     recommend=recommendation(option)
#     for i in recommend:

#         st.write(i)
