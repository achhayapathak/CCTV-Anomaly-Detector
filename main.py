# importing required libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile

# Function to predict anomaly in the video
def predict_anomaly(video):
    # Loading the Model
    model=tf.keras.models.load_model("./Model_10s_80_acc")
    
    #Calculating Frames
    total_frames =  int (video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames<150 or total_frames>1000:
        return -1
    
    x = [] 

    for i in range(1,total_frames//150 +1):
        success,image = video.read()

        count=1
        while count<(i-1)*150:
            try:
                success,image = video.read()
                count+=1
            except:
                pass

        j=0
        y=[]
        while success and j<290:
            try:
                img = cv2.resize(image,(64,64))
                j+=1
                y.append(img)
            except:
                pass
            success,image = video.read()
        if j==290:
            x.append(y)

    # Creating numpy array of the data
    x=np.array(x)

    # Predicting through model
    pred=model.predict(x)

    # If at any instance anomaly is detected, return true
    for l in pred:
        if l[1]<l[0]:
            return True
    return False


def main():
    
    st.title("Anomaly Detection")

    # Take input from user, only mp4 accepted
    file = st.file_uploader("Upload a file", type = "mp4")
    show_file = st.empty()

    # Display the file if available
    if not file:
        show_file.info("Please upload a video")
        return
    else:
        show_file.video(file)
    
    # Reading data into a temporary file so that we can use cv2 on it
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    video = cv2.VideoCapture(tfile.name)

    # Function call to predict anomaly
    prediction = predict_anomaly(video)
    
    if(prediction == -1)
        st.write("Please enter a video of length 10 second to 1 minute")
    else
    # Display the output
        if(st.button('Predict')):
            if(prediction):
                st.error("Anomaly Detected!!")
            else:
                st.success("Normal Video")

    file.close()


if __name__ == "__main__":
    main()
