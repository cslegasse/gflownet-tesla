import time
import os
import av
import numpy as np
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, LlavaProcessor, VideoLlavaForConditionalGeneration
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
llava_model_name = "LanguageBind/Video-LLaVA-7B-hf"
llava_processor = LlavaProcessor.from_pretrained(llava_model_name)
llava_model = VideoLlavaForConditionalGeneration.from_pretrained(llava_model_name).to(device)

data_dir = "data"
video_dir = "videos/"
question_file = os.path.join(data_dir, "questions.csv")
output_file = os.path.join(data_dir, "results.csv")

# Load API Keys
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=gemini_api_key)
openai_client = OpenAI(api_key=openai_api_key)

def read_video_frames(video_path, key_frames=True):
    """ Extracts key frames or the full video for processing """
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = [0, total_frames - 1] if key_frames else np.arange(0, total_frames, total_frames / 8).astype(int)
    frames = [frame.to_ndarray(format="rgb24") for i, frame in enumerate(container.decode(video=0)) if i in indices]
    return frames

def generate_subquestions(question):
    """ Generate sub-questions using LLAVA """
    prompt = f"Break down the following question into relevant sub-questions: '{question}'"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text.split("\n")

def determine_needed_context(video_path, question):
    """ Ask LLAVA whether the full video or key frames are needed """
    prompt = f"For the question: '{question}', can it be answered using only the first and last frame, or is the full video context needed?"
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return "full video" in response.text.lower()

def detect_objects_openai(frame):
    """ Use OpenAI vision model for object detection """
    response = openai_client.image.create(file=frame, model="gpt-4-vision")
    return response

def generate_llava_summary(video_path):
    """ Uses LLAVA to generate a reasoning summary for the video """
    video_frames = read_video_frames(video_path, key_frames=False)
    inputs = llava_processor(videos=video_frames, return_tensors="pt").to(device)
    with torch.no_grad():
        summary_ids = llava_model.generate(**inputs, max_new_tokens=80)
    return llava_processor.batch_decode(summary_ids, skip_special_tokens=True)[0]

def main():
    questions_df = pd.read_csv(question_file, dtype={"id": str})
    results = []
    
    for _, row in questions_df.iterrows():
        video_id = row['id'].zfill(5)
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        question = row['question']
        
        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found.")
            continue
        
        print(f"Processing Video {video_id}...")
        subquestions = generate_subquestions(question)
        full_video_needed = determine_needed_context(video_path, question)
        
        if full_video_needed:
            llava_summary = generate_llava_summary(video_path)
        else:
            video_frames = read_video_frames(video_path, key_frames=True)
            llava_summary = detect_objects_openai(video_frames[-1])
        
        prompt = f"Subquestions: {subquestions}\nVideo Reasoning: {llava_summary}\nQuestion: {question}\nState your final answer as: The correct answer choice is X."
        gemini_response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        
        results.append({"id": video_id, "question": question, "answer": gemini_response.text})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
