import time
import av
import numpy as np
import os
import cv2
import torch
import pandas as pd
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel, VideoLlavaForConditionalGeneration, VideoLlavaProcessor
)
import google.generativeai as genai
from dotenv import load_dotenv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load LLAVA VLM model
llava_model_name = "LanguageBind/Video-LLaVA-7B-hf"
llava_processor = VideoLlavaProcessor.from_pretrained(llava_model_name)
llava_model = VideoLlavaForConditionalGeneration.from_pretrained(
    llava_model_name, torch_dtype=torch.float16, device_map="auto"
)

# Directory and file paths
data_dir = "data"
video_dir = "videos/"
question_file = os.path.join(data_dir, "questions.csv")
correct_answers_file = os.path.join(data_dir, "correct50_submission.csv")  # ground truth file: id,answer (e.g., A)
output_file = os.path.join(data_dir, "predictions_comparison.csv")

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        np.ndarray: Decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

@timer_func
def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    # Encode frames using CLIP
    pil_frames = [Image.fromarray(frame) for frame in frames]
    all_frame_features = []
    for batch in range(0, len(pil_frames), 32):  # Process in batches
        batch_frames = pil_frames[batch: batch + 32]
        inputs = clip_processor(images=batch_frames, return_tensors="pt", padding=True)
        frame_tensors = inputs["pixel_values"].to(device)
        with torch.no_grad():
            batch_features = clip_model.get_image_features(frame_tensors)
        all_frame_features.append(batch_features)

    video_features = torch.cat(all_frame_features, dim=0)
    return video_features.mean(dim=0, keepdim=True)

@timer_func
def generate_llava_summary(video_path):
    """Uses LLAVA to generate a textual summary of the video."""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)
    # Prompt format for better results
    prompt = "USER: <video>\nSummarize the key events of this video. ASSISTANT:"
    inputs = llava_processor(text=prompt, videos=video, return_tensors="pt").to(device)
    out = llava_model.generate(**inputs, max_new_tokens=60)
    return llava_processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)

@timer_func
def encode_text(text):
    """Converts text into a CLIP-compatible feature vector."""
    inputs = clip_processor(text=[text], images=None, return_tensors="pt", padding=True, truncation=True)
    return clip_model.get_text_features(inputs["input_ids"].to(device))

class GFlowNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=4):
        super(GFlowNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

@timer_func
def gflow_infer(video_features, text_features):
    """Uses GFlowNet to predict probabilities for each multiple-choice answer."""
    input_dim = video_features.shape[1] + text_features.shape[1]
    gflow_model = GFlowNet(input_dim=input_dim).to(device)

    combined_features = torch.cat([video_features, text_features], dim=1)
    predicted_logits = gflow_model(combined_features)
    probabilities = torch.softmax(predicted_logits, dim=1)

    return probabilities.argmax().item(), probabilities.detach().cpu().numpy()

@timer_func
def get_llm_response(llava_summary, clip_embeddings, probabilities, question_text, question_id):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found in .env file")

    genai.configure(api_key=api_key)

    prob_str = ", ".join([f"Choice {chr(65 + i)}: {prob:.4f}" for i, prob in enumerate(probabilities.flatten())])
    prompt = (
        f"Video Summary from LLAVA: {llava_summary}\n\n"
        f"Text CLIP Embeddings: {clip_embeddings}\n\n"
        f"Multiple-Choice Probabilities from GFlowNet: {prob_str}\n\n"
        f"Question: {question_text}\n\n"
        "State your final answer as: The correct answer choice is X."
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

@timer_func
def main():
    questions_df = pd.read_csv(question_file, dtype={"id": str})
    correct_answers_df = pd.read_csv(correct_answers_file, dtype={"id": str})

    predictions, probabilities_list, gemini_answers = [], [], []
    processed_ids, processed_questions = [], []

    for _, row in questions_df.iterrows():
        video_id = f"{int(row['id']):05d}"
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        question_id = row["id"]
        question_text = row["question"]

        if not os.path.exists(video_path):
            print(f"Warning: Video {video_path} not found.")
            continue

        print(f"Processing Video {video_id}...")

        # Extract Video Features
        video_features = extract_video_features(video_path)

        # Generate LLAVA Summary
        llava_summary = generate_llava_summary(video_path)

        # Encode Question
        text_features = encode_text(question_text)

        # Get Probabilities from GFlowNet
        predicted_index, probabilities = gflow_infer(video_features, text_features)
        predicted_answer = chr(65 + predicted_index)  # Convert index to letter (A, B, C, D)

        # Get Final Answer from Gemini Flash
        gemini_response = get_llm_response(llava_summary, video_features, probabilities, question_text, question_id)

        print(f"LLAVA Summary: {llava_summary}")
        print(f"Question: {question_text}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Probabilities: {probabilities}")
        print(f"Gemini Response: {gemini_response}")

        # Append results for saving later
        processed_ids.append(row["id"])
        processed_questions.append(question_text)
        predictions.append(predicted_answer)
        probabilities_list.append(probabilities)
        gemini_answers.append(gemini_response)

    # Format probabilities for each question (e.g., "Choice A: 0.1234, Choice B: 0.4567, ...")
    formatted_probs = []
    for prob in probabilities_list:
        prob_str = ", ".join([f"Choice {chr(65 + i)}: {p:.4f}" for i, p in enumerate(prob.flatten())])
        formatted_probs.append(prob_str)

    # Create a DataFrame with the collected results
    results_df = pd.DataFrame({
        "id": processed_ids,
        "question": processed_questions,
        "predicted_answer": predictions,
        "gemini_response": gemini_answers,
        "probabilities": formatted_probs
    })

    # Merge with the ground truth answers (file now contains id and answer, e.g., "A")
    results_df = results_df.merge(correct_answers_df, on="id", how="left")
    results_df.rename(columns={"answer": "ground_truth_answer"}, inplace=True)

    # Save the results DataFrame to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()
