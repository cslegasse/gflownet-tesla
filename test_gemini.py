import cv2
import time
import os
import re
import enum

from dotenv import load_dotenv
import pandas as pd
from google import genai
from pydantic import BaseModel

# import google.generativeai as genai
# from google.colab import userdata
load_dotenv()
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

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer_func
def preprocess_video(vid_dir, video_num, out_dir):
    # Input video file
    # video_path = "00044.mp4"  # Change this to your video file
    video_path = f"{video_num:05}.mp4"
    video_path = os.path.join(vid_dir, video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract filename without extension
    output_folder = out_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    first_frame_path = os.path.join(vid_dir, f"{video_name}_first.png")
    last_frame_path = os.path.join(vid_dir, f"{video_name}_last.png")

    if os.path.exists(first_frame_path) and os.path.exists(last_frame_path):
        print(f"First and last frames already exist for {video_name}. Skipping extraction.")
    else:
        cap = cv2.VideoCapture(video_path)
        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Read the first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to first frame
        ret, first_frame = cap.read()
        if ret:
            cv2.imwrite(first_frame_path, first_frame)
            print(f"Saved first frame as {first_frame_path}")

        # Read the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)  # Go to last frame
        ret, last_frame = cap.read()
        if ret:
            cv2.imwrite(last_frame_path, last_frame)
            print(f"Saved last frame as {last_frame_path}")

        cap.release()

    return first_frame_path, first_frame_path, last_frame_path

@timer_func
def upload_to_gemini(video_path, first_frame_path, last_frame_path):
    # Upload the video file to Gemini
    my_file = client.files.upload(file=video_path)
    last_frame = client.files.upload(file=first_frame_path)
    first_frame = client.files.upload(file=last_frame_path)
    return my_file, first_frame, last_frame

@timer_func
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

@timer_func
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

@timer_func
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
    results = []  # Store results for all videos

    for video_num in range(1, 252):
        # TODO: remove this block when ready
        if video_num > 50:
            results.append({"id": f"{video_num:05d}", "answer": "E"})
            continue

        test = get_question_and_answers(video_num)
        print(test)

        if test is None:
            print(f"Skipping video {video_num} due to missing question.")
            results.append({"id": f"{video_num:05d}", "answer": "Unknown"})  # Add "Unknown" for missing questions
            continue

        question, answers = test
        context_local_paths = preprocess_video("videos", video_num, "output")
        video_contexts = upload_to_gemini(video_path=context_local_paths[0], first_frame_path=context_local_paths[1], last_frame_path=context_local_paths[2])
        prompt1, resp1 = prompt_get_observables(video_contexts, question, answers)
        resp2 = prompt_get_answer(video_contexts, question, answers, prompt1, resp1)

        answer_match = re.search(r'"answer":\s*"([A-E])"', resp2)  # Adjust regex if needed
        if answer_match:
            predicted_answer = answer_match.group(1)
        else:
            predicted_answer = "Unknown"  # Handle cases where the answer isn't found

        results.append({"id": f"{video_num:05d}", "answer": predicted_answer})

    # Create a DataFrame from all results
    output_df = pd.DataFrame(results)

    # Save to CSV
    output_df.to_csv("output/submission.csv", index=False)