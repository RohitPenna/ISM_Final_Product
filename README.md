# ISM_Final_Product

Interactive Chatbot with Real-Time Emotional Analysis/Integration

## Usage

- Clone this repository
- Enter the commands:
  - Make a virtual environment: `python -m venv .venv`
  - Activate it (for Windows command prompt): `.venv\Scripts\activate`

### User Stories

- [ ] A user would like to be able to start up an app and talk to a chatbot that can percieve their emotions in real-time.

### Technology Stack

- Python
- Open CV for webcam image processing
- Streamlit or Gradio
- Huggingface Endpoints for emotion detecting pre-trained models
- OpenAI large language models (Chat GPT)
- Hypermodern Python Coding Standards (<https://cookiecutter-hypermodern-python.readthedocs.io/en/2021.3.14/guide.html>)

### Inspiration Repos

- <https://github.com/hwchase17/langchain>
- <https://github.com/omar178/Emotion-recognition>
- <https://github.com/gradio-app/hub-emotion-recognition>
- <https://github.com/joeychrys/streamlit-chatGPT/blob/master/src/main.py>

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
