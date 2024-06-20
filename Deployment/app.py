import streamlit as st
from PIL import Image
import cv2
import supervision as sv
from time import sleep
from datetime import datetime
from ultralytics import YOLO
import io, os
import subprocess
import numpy as np

# Process frame function
def process_frame(frame: np.ndarray, _) -> np.ndarray:
    vid_results = model(frame, imgsz=1040)[0]

    scores = vid_results.boxes.conf.cpu().numpy()
    if len(scores) > 0:
      detections = sv.Detections.from_ultralytics(vid_results)[int(np.argmax(scores))]
      box_annotator = sv.BoundingBoxAnnotator()
      label_annotator = sv.LabelAnnotator()
      labels = [f"{name['class_name']} {confidence:0.2f}" for _, _, confidence, class_id, _,name in detections]

      # creating the bounding boxes
      annotated_frame = box_annotator.annotate(
        scene=frame,
        detections=detections
      )
      # labeling the bounding box
      frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections, labels=labels
      )
    else:
      pass

    return frame

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

# load model
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
      if len(scores) > 0:
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
        st.write('You have either not uploaded a turtle Image or the turtle Image is Not Clear. Please upload a clear turtle Image that is of good resolution! Your Image should cover the entire shape of the turtle.')

else:
  st.markdown('Upload your video, we will process it, and detect sea turtle face. Takes Longer!')

  # Upload file
  uploaded_video = st.file_uploader('Upload Video', type=['mp4'])

  if uploaded_video != None:
    ts = datetime.timestamp(datetime.now())
    # imgpath = os.path.join('data/', str(ts)+uploaded_video.name)
    # outputpath = os.path.join('data/', os.path.basename(imgpath))
    imgpath = os.path.join(str(ts)+uploaded_video.name)
    outputpath = os.path.join(os.path.basename(imgpath))

    with open(imgpath, mode='wb') as f:
        f.write(uploaded_video.read())  # save video to disk

    st_video = open(imgpath, 'rb')
    video_bytes = st_video.read()
    print('-----------')
    print(imgpath)
    print(outputpath)
    st.video(video_bytes)
    st.write("Uploaded Video")
    if st.button('Detect Turtle Face in Video'):

      sv.process_video(source_path=imgpath, target_path=f"result.mp4", callback=process_frame)

      # Define the command
      command = "ffmpeg -i result.mp4 -vcodec libx264 final_res.mp4"

      # Run the command
      subprocess.run(command, shell=True)

      vid_file = open('final_res.mp4', 'rb')
      vid_bytes = vid_file.read()
      st.header('Detected Video')
      st.video(vid_bytes)
