
import numpy as np
import av
import mediapipe as mp
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import pickle
import streamlit as st
import speech_recognition as sr
import matplotlib.pyplot as plt
from easygui import buttonbox
import os
from PIL import Image
from itertools import count
import tkinter as tk
import string
import time

#TITLE USING STREAMLIT
st.title("SIGN LANGUAGE TRANSLATOR")


selected_option = st.selectbox('Choose',('sign to text','audio to sign'))


if selected_option == 'sign to text':
    filename = 'finalized_model_n_rf.sav'
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


    def process(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_Lms in results.multi_hand_landmarks:
                a = np.array([])
                b = np.array([])
                for id, lm in enumerate(hand_Lms.landmark):
                    # print(id, lm)
                    h, w, c = image.shape
                    a = np.append(a, np.array(int(lm.x * w)))
                    b = np.append(b, np.array(int(lm.y * h)))
                    # print(id, cx, cy)
                a = np.append(a, b)
                a = a.reshape(1, -1)
                mp_drawing.draw_landmarks(
                    image,
                    hand_Lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.rectangle(image, (150, 60), (420, 450), (0, 0, 0), 2)
                # St = list(string.ascii_uppercase)
                #ai = int(loaded_model.predict(a))
                if loaded_model.predict(a) == 0:
                    cv2.putText(image, str("V"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
                else:
                    cv2.putText(image, str("not V"), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
        return image

#To  establish a peer-to-peer connection - webrtc is used to connect users in real time .
# WebRTC connections can't run without a server
#so a stun server is used to connect in real time
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


class VideoProcessor:
    def recv(self, frame):
        #image frame obtained is converted to numpy array
        img = frame.to_ndarray(format="bgr24")
        img = process(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

 # webrtc streaner is Streamlit component which deals with video and audio real-time I/O through web browsers.
#The key argument is a unique ID in the script to identify the component instance.
webrtc_ctx = webrtc_streamer(
        key="WYH",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

if selected_option == 'audio to sign':
    start_button = st.button("Start Microphone Input")
    if start_button:
        def func():
            r = sr.Recognizer()
            arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            # source = microphone by which we give input
            with sr.Microphone(device_index=1) as source:

                # r.adjust_for_ambient_noise(source)
                i = 0
                while True:
                    st.write('Say something')
                    # to remove the backgroundnoise
                    r.adjust_for_ambient_noise(source)
                    # audio= the audio signal which we speak
                    audio = r.listen(source)
                    try:
                        # speech to text conversion
                        a = r.recognize_google(audio)
                        st.write("you said " + format(a))
                        for c in string.punctuation:
                            a = a.replace(c, "")

                        if (a.lower() == 'goodbye' or a.lower() == 'good bye' or a.lower() == 'bye'):
                            st.write("oops!Time To say good bye")
                            break

                        else:
                            for i in range(len(a)):
                                if a[i] in arr:


                                    ImageAddress = 'letters/' + a[i] + '.jpg'
                                    ImageItself = Image.open(ImageAddress)
                                    # convert the image in numpy array format to plot
                                    ImageNumpyFormat = np.asarray(ImageItself)
                                    st.image(ImageNumpyFormat, width=200, caption=a[i])
                                    time.sleep(0.2)
                                else:
                                    continue

                    except:
                        st.write(" ")
                    plt.close()


        while 1:
            image = "signlang.jpeg"
            msg = "HEARING IMPAIRMENT ASSISTANT"
            choices = ["Live Voice", "All Done!"]
            # to select either of the choices given
            reply = buttonbox(msg, image=image, choices=choices)
            # if live audio is selected then go to func else exit
            if reply == choices[0]:
                func()
            if reply == choices[1]:
                quit()



