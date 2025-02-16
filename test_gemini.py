import datetime
import cv2
import os
import re
import enum
import requests
import pathlib

import pandas as pd
from typing import Literal
from google import genai
from pydantic import BaseModel

# import google.generativeai as genai
# from google.colab import userdata
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)

class Observable(BaseModel):
  timestamp: str
  event: list[str]

class AnswerLiteral(enum.Enum):
  A = "A"
  B = "B"
  C = "C"
  D = "D"
  E = "E"

class Answer(BaseModel):  
  answer: AnswerLiteral
  reasoning: str

def preprocess_video(vid_dir, video_num):
    # Input video file
    # video_path = "00044.mp4"  # Change this to your video file
    video_path = f"{video_num:05}.mp4"
    video_path = os.path.join(vid_dir, video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract filename without extension
    output_folder = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get total frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to first frame
    ret, first_frame = cap.read()
    if ret:
        first_frame_path = os.path.join(output_folder, f"{video_name}_first.png")
        cv2.imwrite(first_frame_path, first_frame)
        print(f"Saved first frame as {first_frame_path}")

    # Read the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)  # Go to last frame
    ret, last_frame = cap.read()
    if ret:
        last_frame_path = os.path.join(output_folder, f"{video_name}_last.png")
        cv2.imwrite(last_frame_path, last_frame)
        print(f"Saved last frame as {last_frame_path}")

    cap.release()

    my_file = client.files.upload(file=video_path)
    last_frame = client.files.upload(file=first_frame_path)
    first_frame = client.files.upload(file=last_frame_path)
    return [my_file, first_frame, last_frame]

def prompt_get_observables(video_contexts, question, answers):
    # Create the prompt.
    # question = "Why is it appropriate for ego to remain stopped?"
    # answers = " A. Waiting for right of way. B. For a traffic light. C. For a stop sign. D. For a pedestrian."

    prompt = f"""
    Given this question and these possible answers, a relevant video, and the first and last frames from that video, 
    1. Brainstorm a list of observables that you think will be essential to answering the question. 
    2. Generate an event log that describes the presence of the observables. The timestamp for each event should be 00:00 (seconds:milliseconds)
    ===
    Question: {question}
    Answers: {answers}
    IMPORTANT: Consider the whole video before you start identifying observables, because some observables may only be visible later in the video.
    IMPORTANT: The potential answers may refer to nonexistent observables, incorrect facts, or made up scenarios. Rely more on the question than the answers to generate the observables event log.
    """
    # Set the model to Gemini Flash.
    # model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    # Make the LLM request.
    print("Making LLM inference request...")
    # response = model.generate_content([prompt, video_file],
    #                                   request_options={"timeout": 600})
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=[
            prompt, 
            *video_contexts
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[Observable],
        },
        # config={"timeout": 10}
    )
    print(response.text)
    # print(response.text)
    return prompt, response.text

def prompt_get_answer(video_contexts, question, answers, first_prompt, first_resp):
    # Create the prompt.
    second_prompt = f"""
    Given the following question, answers, and a time-ordered event log of what's happening in a video,
    1. Reason through what's possible and impossible temporally based on the observations.
    2. Give me the most appropriate answer in the list of potential answers.
    ===
    Question: {question}
    Answers: {answers}
    Observations: {first_resp}
    """
    # Set the model to Gemini Flash.
    # model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    # Make the LLM request.
    print("Making LLM inference request...")
    # response = model.generate_content([prompt, video_file],
    #                                   request_options={"timeout": 600})
    response = client.models.generate_content(
        model='gemini-2.0-flash', 
        contents=[
            first_prompt,
            *video_contexts,
            # my_file,
            # first_frame,
            # last_frame,
            second_prompt
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': Answer,
        },
        # config={"timeout": 10}
    )
    print(response.text)
    # print(response.text)
    return response.text

def get_question_and_answers(question_num, file_path="data/questions.csv"):
    """
    Fetches the question and multiple-choice answers based on the given question ID.
    
    Parameters:
        question_id (int): The ID of the question to retrieve.
        file_path (str): The path to the CSV file containing the questions.

    Returns:
        dict: A dictionary with 'question' and 'answers' (list of answer choices).
    """
    # Load the CSV file
    df = pd.read_csv(file_path, 
                     dtype={'id': 'Int32', 'question': 'object'})
    
    # Find the question matching the given ID
    # question_id = f"{question_num:05}"
    question_row = df[df["id"] == question_num]
    
    if not question_row.empty:
        question_text = question_row["question"].values[0]
        
        # Split the question and the multiple-choice options
        match = re.split(r'(?=\s*A\.)', question_text, maxsplit=1)  # Splitting at "A." assuming answers start there
        
        if len(match) == 2:
            question_part = match[0].strip()
            answers = match[1].strip()
            # answers = re.split(r'(?=\s*[A-D]\.)', answers_part)  # Splitting answers by "A.", "B.", etc.
            # answers = [ans.strip() for ans in answers]
        else:
            question_part = question_text.strip()
            answers = []  # No multiple-choice answers found

        return question_part, answers
    else:
        # return {"question": "Question not found.", "answers": []}
        return None

if __name__ == '__main__':
    video_num = 5
    test = get_question_and_answers(video_num)
    print(test)

    question, answers = test
    video_contexts = preprocess_video("../tesla-real-world-video-q-a/videos/videos/", video_num)
    prompt1, resp1 = prompt_get_observables(video_contexts, question, answers)
    resp2 = prompt_get_answer(video_contexts, question, answers, prompt1, resp1)
    

