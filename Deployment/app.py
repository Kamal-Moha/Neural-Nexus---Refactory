import streamlit as st
from PIL import Image
import cv2
import supervision as sv
from time import sleep
from ultralytics import YOLO
import io
import numpy as np

st.set_page_config(page_title= 'Sea Turtle Face Detector')

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.header('Sea Turtle Face Detector', divider='rainbow')
st.sidebar.title('Settings')
st.sidebar.markdown('''**:blue-background[Step 1]**''')
# task options
task_type = st.sidebar.selectbox(
    "Select Task: Do you want to detect turtle face for an Image or for a Video??",
    ["Image", 'Video']
)
st.markdown('''**:blue-background[Step 2]**''')

model = YOLO('Deployment/best.pt')

if task_type == 'Image':
  st.markdown('Simply upload a picture of a sea turtle, we will detect the face of the turtle by drawing bbox')
  # Upload file
  img_uploaded = st.file_uploader('Upload Image', type=['png', 'jpg', 'webp'])
  if img_uploaded is not None:
    image = Image.open(img_uploaded)
    st.title('Original Image')
    # To read file as bytes:
    file_img = st.image(img_uploaded)
    if st.button('Detect Turtle Face'):
      with st.spinner('Wait for it...'):
          sleep(5)
      result = model(image)[0]
      scores = result.boxes.conf.cpu().numpy()
      if len(scores) <= 0:
            st.write('Image is Not Clear. Please upload an Image that is of good resolution! Your Image should cover the entire shape of the turtle.')
      else:
            detections = sv.Detections.from_ultralytics(result)[int(np.argmax(scores))]
            
            box_annotator = sv.BoundingBoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            labels = [
            f"{name['class_name']} {confidence:0.2f}"
            for _, _, confidence, class_id, _,name
            in detections]
            
            # creating the bounding boxes
            annotated_frame = box_annotator.annotate(
            scene=image.copy(),
            detections=detections
            )
            
            # labeling the bounding box
            label_annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections, labels=labels
            )
            # show image
            # sv.plot_image(image=label_annotated_frame, size=(16, 16))
            st.title('Detected Image')
            st.image(label_annotated_frame)


else:
  st.markdown('Upload your video, we will process it, and detect sea turtle face. Takes Longer!')

  # Upload file
  uploaded_file = st.file_uploader('Upload Video', type=['mp4'])

  if uploaded_file is not None:

    # image = Image.open(uploaded_file)

    st.title('Original Image')

    # To read file as bytes:
    # file_img = st.image(uploaded_file)

    # g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
    temporary_location = "testout_simple.mp4"

    with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
        out.write(g.read())  ## Read bytes into file

    st.video('testout_simple.mp4')
    print('-------------')





