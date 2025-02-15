import cv2
import torch
import pandas as pd
import numpy as np
import clip
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print(torch.cuda.is_available())

# Load language model and tokenizer
llm_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
language_model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

# Load CLIP for vision-language encoding
clip_model, preprocess = clip.load("ViT-B/32", jit=False, device = "cuda" if torch.cuda.is_available() else "cpu")

def extract_video_features(video_path):
    """Extracts key frames and detects objects."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # Check if any frames were extracted
    if len(frames) == 0:
        raise ValueError(f"No frames were extracted from the video at {video_path}")

    return frames

def encode_text(text):
    """Encodes text using CLIP model."""
    text_encoded = clip.tokenize([text]).to("cuda")
    with torch.no_grad():
        text_features = clip_model.encode_text(text_encoded)
    return text_features

def encode_video(frames):
    """Encodes video frames using CLIP."""
    if len(frames) == 0:
        raise ValueError("No frames provided to encode_video.")
    
    # Convert each frame to a PIL Image before preprocessing
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    frame_tensors = torch.stack([preprocess(frame).to("cuda") for frame in pil_frames])
    with torch.no_grad():
        video_features = clip_model.encode_image(frame_tensors)
    return video_features.mean(dim=0)  # Aggregate features

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
        x = self.softmax(self.fc3(x))
        return x

def gflow_infer(video_features, text_features):
    """Implements GFLOW Net reasoning to select the best answer."""
    input_dim = video_features.shape[0] + text_features.shape[0]
    gflow_model = GFlowNet(input_dim=input_dim)
    combined_features = torch.cat([video_features, text_features], dim=0).unsqueeze(0)
    predicted_answer = gflow_model(combined_features)
    return predicted_answer.argmax().item()

def process_single_sample(video_path, question):
    """Processes a single video-question pair."""
    video_features = encode_video(extract_video_features(video_path))
    text_features = encode_text(question)
    answer_idx = gflow_infer(video_features, text_features)
    return answer_idx

# Load questions dataset
questions_df = pd.read_csv("data/questions.csv")

# Process a single sample
sample_video = "videos/00001.mp4"
sample_question = questions_df.iloc[0]["question"]
predicted_answer = process_single_sample(sample_video, sample_question)
print(f"Predicted Answer: Option {chr(65 + predicted_answer)}")
