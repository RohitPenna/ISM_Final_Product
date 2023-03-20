import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import openai
import os
from decouple import config
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from streamlit_chat import message as st_message


# Initialize OpenAI API
openai.organization = config("OPENAI_ORG_NAME")
openai.api_key = config("OPENAI_API_KEY")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # vision processing
        flipped = img[:, ::-1, :]

        # model processing
        im_pil = Image.fromarray(flipped)
        results = st.model(im_pil, size=112)
        bbox_img = np.array(results.render()[0])

        return av.VideoFrame.from_ndarray(bbox_img, format="bgr24")


# # Define function to detect emotions using OpenAI API
# def detect_emotions(image):
#     # Resize image to 256x256
#     resized_image = cv2.resize(image, (256, 256))
#     # Convert image to bytes
#     image_bytes = cv2.imencode('.jpg', resized_image)[1].tobytes()
#     # Call OpenAI API to detect emotions
#     response = openai.Completion.create(
#         engine="davinci",
#         prompt=f"Display some emotions",
#         max_tokens=50,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#     # Extract emotions from OpenAI API response
#     emotions = response.choices[0].text.strip()
#     return emotions

# state management initialization
def initialize_state():
    if "num_prompts_user_sent" not in st.session_state:
        st.session_state.num_prompts_user_sent = 0

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


#TODO dropmenu for initial prompt options
# initial_prompt = ("""You are an AI chatbot on a website that is wired to a live webcam that is recording the user's emotions. 
#     You will receive a series of prompts from said user and information about their emotion extracted from their face while they read your prompt.
#     Your task is to make them feel good and happy. So take their feedback in consideration when constructing happy things to say.
#     Here is their first prompt and emotional status, respond to them to make them feel happy:""")

def dispatch_prompt():
    initial_prompt = ("""I want you to act as a general human companion. 
    You are an AI chatbot wired to a web application user interface and a user will submit general comments to you. 
    Your job is to provide deep diving questions about this person and build a model of who they are in order to entertain them and provide them with companionship.
    You are always meant to be positive and upbeat and to never let the conversation die so you must always ask a thought provoking question to keep the user engaged.
    The conversation history will be provided in every prompt and you will be prepended with `'AI COMPANION':` and the user's responses will be labelled `'USER':` 
    Provide a completion of the following conversation:

    """)

    
    if st.session_state.num_prompts_user_sent == 0:
        full_prompt = f"{initial_prompt} 'USER': {st.session_state.input_text}\n"
        full_prompt += f"\n'AI COMPANION': "
    else:
        user_prompt = st.session_state.input_text

        # construct the whole conversation.
        
        full_prompt = initial_prompt
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                full_prompt += f"'USER': {msg.message}"
            else:
                full_prompt += f"'AI COMPANION': {msg.message}"

        full_prompt += f"\n'USER': {user_prompt}"
        full_prompt += f"\n'AI COMPANION': "

    print(f"{st.session_state.num_prompts_user_sent=}")
    print(f"{full_prompt=}")
    
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": full_prompt}
                        ]
                    )
    st.session_state.chat_history.append({'message':st.session_state.input_text, "is_user":True})
    st.session_state.chat_history.append({'message':response.choices[0].message.content, "is_user": False})
    st.session_state.num_prompts_user_sent +=1


def run_app():
    
    # Set page title
    st.set_page_config(page_title="Web Camera Emotion Detector and Chatbot")

    # Set page layout
    col1, col2 = st.columns([2, 1])


    initialize_state()

    # Add web camera view port to page layout
    with col1:
        st.header("AI ChatBot")

        # start text input game.

        st.text_input("Talk to ChatGPT Here!", key='input_text', on_change=dispatch_prompt)

        # Display response
        with st.container():
            st.text('Chat History:')
            for chat in st.session_state.chat_history:
                st_message(**chat)
    # Add chatbox input/output display to page layout
    with col2:
        st.header("Live Emotion Detector")
        # Initialize video feed
        camera_stream = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    

    # "st.session_state object", st.session_state

# Run the app
if __name__ == '__main__':
    run_app()