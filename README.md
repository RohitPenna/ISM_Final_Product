# ISM_Final_Product

Interactive Chatbot with Real-Time Emotional Analysis/Integration

## Usage

- Clone this repository
- Enter the commands:

  - Make a virtual environment: `python -m venv .venv`
  - Activate it (for Windows command prompt): `.venv\Scripts\activate`
  - `pip install requirements.txt -r`
  - Open an OpenAI account at `https://platform.openai.com/account/org-settings`
  - Add a .env file to the top level directory with the following environment variables.

    ```
    OPENAI_API_KEY=xxxx
    OPENAI_ORG_NAME=xxxxx
    ```

  - `cd src/`
  - `streamlit run app.py`

### User Stories

- [ ] A user would like to be able to start up an app and talk to a chatbot that can percieve their emotions in real-time.

### Technology Stack

- Python
- Open CV for webcam image processing
- Streamlit or Gradio
- Huggingface Endpoints for emotion detecting pre-trained models
- OpenAI large language models (Chat GPT)
- Hypermodern Python Coding Standards (<https://cookiecutter-hypermodern-python.readthedocs.io/en/2021.3.14/guide.html>)

### Inspiration Repos/Articles

- <https://github.com/hwchase17/langchain>
- <https://github.com/omar178/Emotion-recognition>
- <https://github.com/gradio-app/hub-emotion-recognition>
- <https://github.com/joeychrys/streamlit-chatGPT/blob/master/src/main.py>
- <https://blog.streamlit.io/how-to-build-the-streamlit-webrtc-component/>
- <https://github.com/nicolalandro/yolov5_streamlit>
- <https://github.com/nextstep-infotech/Open-AI/blob/main/streamlit_chatGPT_clone.py>
- <https://blog.devgenius.io/facial-emotion-detection-using-aws-rekognition-in-python-69b2da668192>
- <https://github.com/codedev99/fast-face-exp>
- <https://github.com/sanjeevm4788/Face-emotion-detection>
- <https://stackoverflow.com/questions/69439390/streamlit-image-processing>
- <https://github.com/whitphx/streamlit-fesion>
- <https://github.com/whitphx/streamlit-webrtc/blob/56d715a76247cd6303c9ddade5bc2a8e430ddc2f/app.py#L344>

### Huggingface

- <https://huggingface.co/spaces/schibsted/Facial_Recognition_with_Sentiment_Detector>
- <https://huggingface.co/spaces/schibsted/facial_expression_classifier/blob/main/app.py>

### Page Breakdown

- Main page
  - Continuous Webcam Viewport (Potentially with a start and stop button)
  - Text Input/Output textbox with a submit button

### Implementation

- User Data
  - Detected Emotion
  - Prompt/Prompt History
- User Interaction
  - Buttons for camera start and stop
  - Button to submit prompt
- API Calls
  - Emotion Detection Model to get detected emotion
  - Large Language Model to get response to do something with the detected emotion

# TODO

- [ ] Flip the conversation history to show most recent messages first
- [ ] Make the form input chat text clear on change.
- [x] figure out how to continuously grab the video image to pipe into another emotion detection api
- [ ] figure out how to continuously update state with the emotion dictionary and wire it up to the chatbot prompt with the average since last updated.
