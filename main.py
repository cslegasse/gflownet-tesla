import cv2
import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, CLIPProcessor, CLIPModel

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load language model and tokenizer
llm_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
language_model = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

# Load CLIP for vision-language encoding
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


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
    inputs = processor(text=[text], images=None, return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(inputs["input_ids"].to(device))
    return text_features


def encode_video(frames, batch_size=32):
    """Encodes video frames using CLIP with batch processing."""
    if len(frames) == 0:
        raise ValueError("No frames provided to encode_video.")

    pil_frames = [Image.fromarray(frame) for frame in frames]

    all_frame_features = []
    for i in range(0, len(pil_frames), batch_size):
        batch_frames = pil_frames[i:i + batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt", padding=True)
        frame_tensors = inputs["pixel_values"].to(device)

        with torch.no_grad():  # Disable gradient calculation during inference
            batch_features = model.get_image_features(frame_tensors)
        all_frame_features.append(batch_features)

    # Concatenate the features from all batches
    video_features = torch.cat(all_frame_features, dim=0)
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
    input_dim = video_features.shape[1] + text_features.shape[1]
    gflow_model = GFlowNet(input_dim=input_dim).to(device)
    video_features = video_features.to(device)
    text_features = text_features.to(device)
    combined_features = torch.cat([video_features, text_features], dim=1)
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
