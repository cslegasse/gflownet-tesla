import time
import os
import re
import enum

from dotenv import load_dotenv
import pandas as pd
from google import genai
from pydantic import BaseModel

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

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
def get_video_assets_from_cache(uploaded_files, video_num):
    video_info = uploaded_files.get(video_num)
    if video_info:
        return (video_info['video'], video_info['first_frame'], video_info['last_frame'])
    else:
        print(f"Video {video_num} not found in cache. Uploading...")
        video_file = client.files.upload(file=f"videos/{video_num:05}.mp4")
        first_frame = client.files.upload(file=f"videos/{video_num:05}_first.png")
        last_frame = client.files.upload(file=f"videos/{video_num:05}_last.png")
        return [video_file, first_frame, last_frame]


@timer_func
def prompt_get_observables(video_contexts, question, answers):
    prompt = f"""
    Given this question and these possible answers, a relevant video, and the first and last frames from that video,
    1. Brainstorm a list of observables that you think will be essential to answering the question.
    2. Generate an event log that describes the presence of the observables. The timestamp for each event should have the format "MM:SS".
    ===
    Question: {question}
    Answers: {answers}
    IMPORTANT: Consider the whole video before you start identifying observables, because some observables may only be visible later in the video.
    IMPORTANT: The potential answers may refer to nonexistent observables, incorrect facts, or made up scenarios. Rely more on the question than the answers to generate the observables event log.
    """


    print("Making LLM inference request...")
   
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            *video_contexts,
            prompt
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[Observable],
            'temperature': 0.2,
        },
    )

    print(response.text)
    return prompt, response.text

@timer_func
def prompt_get_answer(video_contexts, question, answers, first_prompt, first_resp):

    second_prompt = f"""
    Given the following question, answers, and a time-ordered event log of what's happening in a video,
    1. Reason through what's possible and impossible temporally based on the observations.
    2. Give me the most appropriate answer in the list of potential answers.
    ===
    Question: {question}
    Answers: {answers}
    Observations: {first_resp}
    """


    print("Making LLM inference request...")
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            *video_contexts,
            first_prompt,
            second_prompt
        ],
        config={
            'response_mime_type': 'application/json',
            'response_schema': Answer,
            'temperature': 0.2,
        },
    )
    print(response.text)
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

    df = pd.read_csv(file_path,
                     dtype={'id': 'Int32', 'question': 'object'})

    question_row = df[df["id"] == question_num]

    if not question_row.empty:
        question_text = question_row["question"].values[0]

        match = re.split(r'(?=\s*A\.)', question_text, maxsplit=1)  

        if len(match) == 2:
            question_part = match[0].strip()
            answers = match[1].strip()
        else:
            question_part = question_text.strip()
            answers = []  

        return question_part, answers
    else:
        return None

if __name__ == '__main__':
    client = genai.Client(api_key=GOOGLE_API_KEY)
    results = []  
    uploaded_files = {item['id']: item for item in pd.read_csv("gemini_cache.csv").to_dict(orient='records')}

    for video_num in range(1, 252):
        # if video_num > 250:
        #     results.append({"id": f"{video_num:05d}", "answer": "E"})
        #     continue

        test = get_question_and_answers(video_num)
        print(test)

        if test is None:
            print(f"Skipping video {video_num} due to missing question.")
            results.append({"id": f"{video_num:05d}", "answer": "Unknown"}) 
            continue

        question, answers = test
        video_contexts = get_video_assets_from_cache(uploaded_files, video_num)
        prompt1, resp1 = prompt_get_observables(video_contexts, question, answers)
        resp2 = prompt_get_answer(video_contexts, question, answers, prompt1, resp1)

        answer_match = re.search(r'"answer":\s*"([A-E])"', resp2)  # Regex
        if answer_match:
            predicted_answer = answer_match.group(1)
        else:
            predicted_answer = "Unknown"  # Handle cases where the answer isn't found (Gemini)

        results.append({"id": f"{video_num:05d}", "answer": predicted_answer})

    # Create a DataFrame from all results
    output_df = pd.DataFrame(results)
    output_df.to_csv("output/submission.csv", index=False)