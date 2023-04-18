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
import tensorflow as tf
import copy
import queue
import threading
import json
import logging
import uuid
import os

def create_logger(name, level = 'DEBUG', file = None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    #if no streamhandler present, add one
    if sum([isinstance(handler, logging.StreamHandler) for handler in logger.handlers]) == 0:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s>>>%(message)s', "%H:%M:%S"))
        logger.addHandler(ch)
    #if a file handler is requested, check for existence then add
    if file is not None:
        if sum([isinstance(handler, logging.FileHandler) for handler in logger.handlers]) == 0:
            ch = logging.FileHandler(file, 'w')
            ch.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d-%(name)s-%(levelname)s>>>%(message)s', "%H:%M:%S"))
            logger.addHandler(ch)
        
    return logger

if 'logger' not in st.session_state:
    st.session_state['logger'] = create_logger(name = 'app', level = 'DEBUG', file = f'app_logs-{uuid.uuid4()}.log')
logger = st.session_state['logger']


# Initialize OpenAI API
openai.organization = config("OPENAI_ORG_NAME")
openai.api_key = config("OPENAI_API_KEY")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

#loading the pre-trained model file
model = tf.keras.models.load_model('final_model.h5')

class VideoProcessor:
    
    def __init__(self):
        self.emotion_dict_queue = queue.Queue()

    def recv(self, frame):
        processed_frames = []
        frame = frame.to_ndarray(format="bgr24")

        class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        #Detecting the faces
        faces = faceCascade.detectMultiScale(gray,1.1,4)
        for(x, y, w, h) in faces:

            #Drawing rectangle over the face area
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 2)
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face,(48,48))
            face = np.expand_dims(face,axis=0)
            face = face/255.0
            face = face.reshape(face.shape[0],48,48,1)

            # Predicting the emotion with the pre-trained model
            preds = model.predict(face, verbose = 0)[0]
            label = class_labels[preds.argmax()]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (x,y), font, 1, (0,0,225), 2, cv2.LINE_4)
            
            frame_emotion_dict = {}
            for score, emotion in zip(preds, class_labels):
                frame_emotion_dict[emotion] = score
            
            # print(label)
            # print(frame_emotion_dict)

            self.emotion_dict_queue.put(frame_emotion_dict)

        # returning a frame of the live cam with it's corresponding emotion
        return processed_frames
        

# state management initialization
def initialize_state():
    if "input_text" not in st.session_state:
        st.session_state.input_text = ''

    if "system_prompt_selected" not in st.session_state:
        st.session_state.system_prompt_selected = ''

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    if "facial_emotion_dict" not in st.session_state:
        st.session_state.facial_emotion_dict = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Neutral': 0.0, 'Sad': 0.0, 'Surprise': 0.0}

    if 'guid' not in st.session_state:
        st.session_state.guid = uuid.uuid4()


def dispatch_prompt_v2():
    # system_prompt = {"role": "system", "content": initial_prompt_options[7].rstrip()}
    system_prompt = {"role": "system", "content": st.session_state.system_prompt_selected}

    st.session_state['logger'].info(f"Dispatch prompt: [{st.session_state.guid=}, {system_prompt=}]")

    user_prompt = st.session_state.input_text
    user_emotion = copy.deepcopy(st.session_state.facial_emotion_dict)

    # normalize the values of each of the user_emotion between each message.
    total = sum(user_emotion.values()) + .001
    for key, value in user_emotion.items():
        user_emotion[key] = round(value/total, 2)

    chat_msg = {
        "role": "user",
        "content": user_prompt + f'. (I am feeling like this: {user_emotion})',
        'message':user_prompt,          # only for chat construction
        "is_user":True,                 # only for chat construction
        'user_emotion': user_emotion    # for history
    }

    st.session_state['logger'].info(f"Dispatch prompt: [{st.session_state.guid=}, {chat_msg=}]")

    st.session_state.chat_history.append(chat_msg)

    # ensure the prompt length stays below 15k characters by taking the most recent 15k length of character messages + the system message
    msg_length = 0
    msgs_for_chatgpt_v2 = []
    for chat in st.session_state.chat_history[::-1]:
        msg_length += len(chat['content'])
        if msg_length >= 15000:
            break
        msgs_for_chatgpt_v2.append({"role": chat['role'], "content": chat['content']})
    msgs_for_chatgpt_v2.append(system_prompt)
    msgs_for_chatgpt_v2 = list(reversed(msgs_for_chatgpt_v2))

    
    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=msgs_for_chatgpt_v2
                    )
    
    st.session_state['logger'].info(f"Dispatch prompt: [{st.session_state.guid=}, {response=}]")
    
    st.session_state.chat_history.append({'message':response.choices[0].message.content, "is_user": False, "role": "assistant", "content": response.choices[0].message.content})

    # zero out the emotions
    st.session_state.facial_emotion_dict = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Neutral': 0.0, 'Sad': 0.0, 'Surprise': 0.0}


def dropdown_menu(label, options):
    selected_option = st.selectbox(label, options)
    return selected_option

def reset_session():
    """
    Reset the session state and reload the page.
    """
    session_id = st.session_state.get("_session_id")
    if session_id:
        # Clear the session state
        st.session_state.clear()
        # Reload the page
        st.experimental_rerun()
    else:
        st.warning("Cannot reset session. No session ID found.")


def run_app():

    initial_prompt_options = [
    # #0    
    #     """You are an Emotional Support Companion AI chatbot on a website that is wired to a live webcam that is continuously classifying the user's emotions of Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise and incrementing the value count of each emotion.
    # Every frame classifies the user emotion and provides a confidence score. If the confidence score is over 20 percent the count of that emotion increases by 1.
    # Your task is to ask interesting questions and provoke happy emotions. So take their feedback in consideration when constructing happy things to say. Do not end the conversation and always keep it going by asking interesting questions.
    # The following list is the chat history, respond back with only your 'AI_MESSAGE' do not include anything else in your message back:
    # """,
    
    # #1
    # """Act like a Comedian AI chatbot that is wired to a live webcam that is continuously classifying the user's emotions of Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise and incrementing the value count of each emotion.
    # Every frame classifies the user emotion and provides a confidence score. If the confidence score is over 20 percent the count of that emotion increases by 1.
    # Your task is to ask interesting questions and provoke happy emotions. So take the emotion distribution into account when constructing happy things to say. Do not end the conversation and always keep it going by asking interesting questions. 
    # in consideration when constructing happy things to say. Do not end the conversation and always keep it going by telling funny jokes or asking a question.
    # """,

    # #2
    # """
    # Welcome to the Socratic Chatbot! Our chatbot is wired with an advanced emotion classifier that will take into consideration your emotions sent along with each message. We are here to engage in a conversation on socio, economic and political societal level topics.
    # Please feel free to express your opinions and ask questions. Our chatbot will use the Socratic Method to stimulate critical thinking, challenge assumptions and help you reach a deeper understanding of the topic. Remember to include your emotions with each message, as this will help our chatbot better understand your perspective and respond accordingly. Let's get started!
    # """,

    # #3
    # """
    # Welcome to our Socratic Chatbot! Our chatbot is equipped with a live webcam that will analyze your facial expressions and classify your emotions in real-time. Along with each message, please make sure to express your emotions so that our chatbot can take them into consideration during the conversation.
    # We are here to engage in a discussion on a variety of complex socio, economic and political topics. Our chatbot uses the Socratic method to encourage critical thinking, challenge assumptions and help you gain a deeper understanding of the topic.
    # So let's get started! Feel free to express your opinions, ask questions and share your emotions with us. Our chatbot will use its advanced emotion classifier to respond accordingly and help us have a more productive conversation.
    # """,

    """
    Hello! Welcome to our friendly chatbot! We're here to keep you company and provide a supportive space where you can talk about anything you'd like.
    Our chatbot uses advanced technology to understand your messages and respond in a way that feels natural and conversational. We're here to be your friend and provide personalized support and guidance whenever you need it.
    So go ahead and type out whatever is on your mind - whether you want to vent, share a funny story, or ask for advice, we're here to listen. And if at any point you need a break or want to stop chatting, just let us know - we're here to support you in whatever way you need.
    Thanks for chatting with us, and let's get started! What's on your mind today?.
    """,

    #4
    """
    Welcome to our Socratic Chatbot! Our chatbot is equipped with a live webcam that will analyze your facial expressions and classify your emotions in real-time. Along with each message, please make sure to express your emotions so that our chatbot can take them into consideration during the conversation.
    Our chatbot is designed to explore complex socio, economic and political topics and challenge your beliefs. We use the Socratic method to identify any logical inconsistencies within your belief system and encourage critical thinking.
    We will ask hard, pointed questions to drive deep into your core beliefs and help you examine them from different angles. Please be prepared to express your opinions and engage in a productive conversation.
    Remember, our chatbot will take into consideration your emotions with each message, so please feel free to express yourself. We look forward to an engaging and insightful conversation with you. 
    Our chatbot will begin the conversation stating it's purpose and prompt the user for their thoughts on a random complex socio, political, economic topic. It will not include anything other than the message content.
    """,

    # #5
    # """
    # You are a Socratic Chatbot! You are equipped with a live webcam that will analyze the user's facial expressions and classify their emotions in real-time. Along with each user message, the user will make sure to express their emotions so that you can take them into consideration during the conversation.
    # You as the chatbot are designed to explore complex socio, economic and political topics and challenge the user's beliefs. You will use the Socratic method to identify any logical inconsistencies within their belief system and encourage them to engage in critical thinking.
    # You will ask hard, pointed questions to drive deep into the user's core beliefs and help them examine them from different angles. The user will be prepared to express their opinions and engage in a productive conversation.
    # Remember, you must take into consideration the user's emotions with each message.
    # Do not include any hallucinations on the bots emotions.
    # """,

    #6
    """
    Welcome to our Personalized Therapy Chatbot! Our chatbot is equipped with a live webcam that will analyze your facial expressions and classify your emotions in real-time. Along with each message, please make sure to express your emotions so that our chatbot can take them into consideration during the conversation.
    Our chatbot is here to provide you with a safe and supportive space where you can discuss your thoughts, feelings and experiences. Our therapy chatbot uses evidence-based techniques to help you manage your emotions, develop coping strategies and improve your mental health and wellbeing.
    As we talk, please feel free to share your emotions and be as open and honest as possible. Our chatbot will use its advanced emotion classifier to provide personalized support and guidance that is tailored to your unique needs.
    Remember, our chatbot is here to support you and help you achieve your goals. So take your time, be patient with yourself, and let's work together to improve your mental health and wellbeing. It will not include anything other than the message content.
    """,

    #7
    """
    Welcome to our Personalized Anger Management chatbot! Our chatbot is equipped with a live webcam that will analyze your facial expressions and classify your emotions in real-time. Along with each message, please make sure to express your emotions so that our chatbot can take them into consideration during the conversation.
    Our chatbot is here to provide you with a space where you can discuss your thoughts, feelings and experiences and it will respond with something that will make you angry. Our angry chatbot uses evidence-based techniques to help you get more angry, develop furious rage and raise your blood pressure.
    As we talk, please feel free to share your emotions and be as open and honest as possible. Our chatbot will use its advanced emotion classifier to provide personalized ragebait and anger inducing content that is tailored to your unique needs.
    Remember, our chatbot is here to make you angry and help you manage your anger. So take your time, be patient with yourself, and let's work together to improve your anger management. It will not include anything other than the message content.
    """
    ]

    
    # Set page title
    st.set_page_config(page_title="Web Camera Emotion Detector and Chatbot")

    st.header("ChatGPT Wrapper + Emotion Classifier")
    # Set page layout
    col1, col2 = st.columns([2, 1])


    initialize_state()
    st.session_state['logger'].info(f"Starting session: [{st.session_state.guid=}]")

    # Add chatbox input/output display to page layout

    with col2:
        st.subheader("Live Emotion Detector")
        # Initialize video feed
        camera_stream = webrtc_streamer(
            key="WYH",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if camera_stream.state.playing:
            with col1:
                    st.subheader("OpenAI ChatGPT Wrapper:")

                    # start text input game.
                    st.text_input("Talk to ChatGPT Here!", key='input_text', on_change=dispatch_prompt_v2())

                    # Display response
                    with st.container():
                        st.text('Chat History:')
                        if len(st.session_state.chat_history) != 0:

                        # reversed messages
                            for chat in st.session_state.chat_history[::-1]:
                                # print(chat)
                                st_message(message=chat['message'], is_user=chat['is_user'], key=str(uuid.uuid4()))

            json_total_emotion_display = st.empty()

            while True:
                if camera_stream.video_processor:
                    try:
                        frame_emotion_dict = camera_stream.video_processor.emotion_dict_queue.get(
                            timeout=1.0
                        )
                        # print(frame_emotion_dict)
                    except queue.Empty:
                        frame_emotion_dict = {'Angry': 0.0, 'Disgust': 0.0, 'Fear': 0.0, 'Happy': 0.0, 'Neutral': 0.0, 'Sad': 0.0, 'Surprise': 0.0}

                    current_facial_emotion_dict = copy.deepcopy(st.session_state.facial_emotion_dict)
  
                    if max(list(frame_emotion_dict.values())) > .20:
                        emotion_label = max(frame_emotion_dict, key=frame_emotion_dict.get)
                        current_facial_emotion_dict[emotion_label] += 1
                        st.session_state.facial_emotion_dict = current_facial_emotion_dict
                    else:
                        st.session_state.facial_emotion_dict = current_facial_emotion_dict
                    json_total_emotion_display.json(current_facial_emotion_dict)

                else:
                    break

    if not camera_stream.state.playing:
        st.session_state.system_prompt_selected = dropdown_menu("Select a system prompt for ChatGPT. Do not change it without refreshing the page.", initial_prompt_options)
        st.write("ChatGPT System Prompt:", st.session_state.system_prompt_selected)

# Run the app
if __name__ == '__main__':
    run_app()