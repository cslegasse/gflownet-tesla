import cv2
import pandas as pd
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")


def preprocess_video(vid_dir, video_num, out_dir):
    # Input video file
    # video_path = "00044.mp4"  # Change this to your video file
    video_path = f"{video_num:05}.mp4"
    video_path = os.path.join(vid_dir, video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[
        0
    ]  # Extract filename without extension
    output_folder = out_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    first_frame_path = os.path.join(vid_dir, f"{video_name}_first.png")
    last_frame_path = os.path.join(vid_dir, f"{video_name}_last.png")

    if os.path.exists(first_frame_path) and os.path.exists(last_frame_path):
        print(
            f"First and last frames already exist for {video_name}. Skipping extraction."
        )
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


def upload_to_gemini(video_path, first_frame_path, last_frame_path):
    my_file = client.files.upload(file=video_path)
    last_frame = client.files.upload(file=first_frame_path)
    first_frame = client.files.upload(file=last_frame_path)
    return my_file, first_frame, last_frame


if __name__ == "__main__":
    client = genai.Client(api_key=GOOGLE_API_KEY)
    uploaded_files = []  # Store results for all videos

    for video_num in range(1, 252):
        # TODO: remove this block when ready
        if video_num > 10:
            break

        context_local_paths = preprocess_video("videos", video_num, "output")
        uploaded_file_assets = upload_to_gemini(
            video_path=context_local_paths[0],
            first_frame_path=context_local_paths[1],
            last_frame_path=context_local_paths[2],
        )
        uploaded_files.append(
            {
                "id": f"{video_num:05d}",
                "video": uploaded_file_assets[0].name,
                "first_frame": uploaded_file_assets[1].name,
                "last_frame": uploaded_file_assets[2].name,
            }
        )

    # Create a DataFrame from all results
    gemini_cache = pd.DataFrame(uploaded_files)

    # Save to CSV
    gemini_cache.to_csv("gemini_cache.csv", index=False)
