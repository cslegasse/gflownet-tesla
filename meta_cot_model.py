from openai import OpenAI
import os
import json
import base64
import csv
import google.generativeai as genai
import re
from PIL import Image
import io
import PIL
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch
import time


# Part1 1: Prompting for detailed sub queries

# Set your OpenAI API Key

genai.configure()

# Function to call OpenAI API
def call_openai_gpt(prompt, model="gpt-4o-mini", imagedata=None):
    
    
    client = OpenAI()
    
    if imagedata: 
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":  prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{imagedata}"},
                        },
                    ],
                }
            ],
        )
   
    else: 
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Ego is an autonomous vehicle navigating the streets, \
                            following road rules, avoiding pedestrians and objects, and responding to traffic \
                            signs, speed limits, and parking restrictions. It must also be aware of nearby vehicles."},
                    {"role": "user", "content": prompt}],
            temperature=0.7)
    return response.choices[0].message.content

# Step 1: Extract Key Visual Features
def extract_key_features(mcq_question, mcq_answers):
    prompt1 = f"""
    "Given the multiple-choice question and answer choices below, analyze what visual features must be detected in the image to answer the question. Use only the provided contextual information and vocabulary related to autonomous vehicle perception."

    Context: Ego is an autonomous vehicle navigating the streets, following road rules, avoiding pedestrians and objects, and responding to traffic signs, speed limits, and parking restrictions. It must also be aware of nearby vehicles.
    Question: {mcq_question}
    Answer Choices: {mcq_answers}
    Vocabulary & Concepts: Road lanes, vehicles, blinkers, lane merging, obstacles, traffic signs, hazardous conditions, motion estimation, object detection, scene segmentation, depth estimation, optical flow.

    "Based on the question and answer choices, list 3-4 key features that must be detected in the image to answer the question correctly."
    
    **Example Output Format:**
    {{
      "key_features": [
        "Presence of traffic signs",
        "Lane markings and their continuity",
        "Positioning of nearby vehicles",
        "Pedestrian presence and movement direction",
        "Traffic light state",
        "Crosswalk presence and pedestrian right-of-way",
        "Speed limit sign detection"
      ]
    }}
    
    """
    
    key_features = call_openai_gpt(prompt1)
    return key_features

# Step 2: Generate Structured Perception Questions
def generate_perception_questions(key_features):
    prompt2 = f"""
    "Given the identified key visual features from the image, generate 2 evaluation questions that focus on object detection, pose estimation, or flow estimation. These questions should help extract the necessary information from the image to solve the original question."

    Identified Features: {key_features}

    Task Type: Object Detection, Pose Estimation, Flow Estimation.
    
    "Generate two to three structured perception questions that an AV system must evaluate from the image for EACH FEATURE. Each question should be framed to extract measurable, structured data relevant to answering the original MCQ question."
    
    **Return the output formatted as follows:**
    {{
      "perception_questions": [
        {{
          "feature": "Presence of traffic signs",
          "questions": [
            "What traffic signs are visible in the image?",
            "Are the traffic signs clear and readable?",
            "Is there a stop sign in the scene?"
          ]
        }},
        {{
          "feature": "Lane markings and their continuity",
          "questions": [
            "Are lane markings clearly visible?",
            "Are any lanes merging in this frame?",
            "Is the ego vehicle within its lane?"
          ]
        }}
      ]
    }}
    """

    perception_questions = call_openai_gpt(prompt2)
    
    print("PERCEPTION QUESTIONS", perception_questions)
    return perception_questions

# Convert String to dictionary
def extract_questions(perception_data):
    questions = []
    for item in perception_data.get("perception_questions", []):
        questions.extend(item.get("questions", []))
        
    print("questions", questions)
    return questions

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Step 3: Meta CoT evaluation 
def validate_features_with_image(image_path, perception_questions, meta_question):
    
    image_data = encode_image(image_path)
    
    # image = PIL.Image.open(image_path)
    # model = genai.GenerativeModel("gemini-1.5-pro-vision")  

    # Construct the prompt
    prompt3 = f"""
    
    Meta question = {meta_question}
    Sub questions = {perception_questions}
    
    "Look at the following list of sub-questions, the meta question, and the image attached. For each sub-question, determine whether:
    1. The features present in the sub-question are present in the image
    2. The sub-question can help solve the meta question. 
    
    ONLY if both questions are answered will it be returned to the user, otherwise it will be disgarded. Otherwise, do not modify the sub-questions. DO NOT MODIFY THE QUESTIONS - only retun the valid questions in a list format.
    
    
    **Ensure that the valid subquestions are returned as a json list!:**
    
    ["Valid Question 1",
     "Valid Question 2", 
     "Valid Question 3"]
     
    """
    
    response = call_openai_gpt(prompt3, model="gpt-4o-mini", imagedata=image_data)
    
    print("RESPONSE", response)

    # json_string = re.sub(r"```json|```", "", response.text).strip()
    # perception_questions_valid = call_openai_gpt(prompt3, model="gpt-o1", imagedata=image_data)
    
    print("DONE")
    return response
# json.loads(json_string)
    
# Part 2: Feature extraction from video 
def analyze_image_with_gemini(image, questions):
    prompt4 = f'''
            Sub questions: {questions}
            
            Look at the following sub questions. Query each of the provided questions on the image provided and answer the question. Provide a clear, concise, and thorough response based on the visual content of the image.
            
            **Return the valid sub-questions formatted in a json dictionary as follows:**
            
            "How many nearby vehicles are detected?" : "There are at least 4 vehicles detected in the image. This includes a car directly in front of the ego vehicle, a pickup truck ahead of that, and two additional vehicles further down the road.",
            "What are the positions of the nearby vehicles relative to the ego vehicle?": "The car is directly ahead. The pickup truck is further ahead."
            
        '''
        
    response = call_openai_gpt(prompt4, model="gpt-4o-mini", imagedata=image)
    # model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
    # response = model.generate_content(contents=[prompt4, image])
    
    print(response)

    json_string = re.sub(r"```json|```", "", response.text).strip()
    
    # print(json_string)
    dic = json.loads(json_string)
    
    combined_string = " ".join(str(value) for value in dic.values())
    
    return combined_string

# Part 3: Video samples - returns frames + last frame
def create_video_subsamples(video_path, questions, frame_interval=35):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break  # Stop when video ends

        frame_count += 1
        if frame_count % frame_interval == 1:  # Extract every 35th frame
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Process the image with Gemini Vision
            results = analyze_image_with_gemini(image, questions)
            # print(f"analysed image {frame_count}")
            
            saved_frames.append(results)

    cap.release()
    # print(f"Extracted {len(saved_frames)} results: {saved_frames}")
    return saved_frames, image

# Part 4: Collate all infromation temporally 
def collate_temporaly(saved_answers):
    prompt5 = f'''
    
    I am going to provide you with a list of descriptions of frames taken sequentially from the same video. Analyze the trends and difference between the responses from each frame and combine that information. 
    
    list of description = {saved_answers}
    
    IF the value stayed constant, describe the value and then state that it remained static/constant throughout the video. ELSE, describe the trend. 
    
    NOTE: INFORMATION FROM THE LAST SLIDE IS THE MOST VALUABLE - it should be weighted the most.
    
    Ensure to mention all points. 
    
    Go through each question sequentially and OUTPUT in a paragraph format. 
    
    BE VERY SHORT AND CONCISE
    
    '''
    
    response = call_openai_gpt(prompt5)
    
    return response
    
# Part 5: Perform Reasoning 
def reasoning_encoding(last_frame, temporal_answer):
    prompt6 = f'''
    Ego is an autonomous vehicle navigating the streets, \
                            following road rules, avoiding pedestrians and objects, and responding to traffic \
                            signs, speed limits, and parking restrictions. It must also be aware of nearby vehicles.
    
    You are given a description of a video taken from inside ego, and the last frame of that video.  Describe the following information: 
    
    
    1. Identify the type of street environment** (highway, city road, alley, parking lot, intersection, crosswalk, etc.).
    2. Locate the ego vehicle in relation to its surroundings**, including:
    - Road lanes (is it centered, merging, or off-lane?)
    - Traffic signs (is it following them?)
    - Other vehicles (is it close to any, merging, stopped?)
    - Pedestrians (are they near or crossing?)
    - Obstacles (potholes, barriers, construction zones, weather conditions)
    3. Check if the ego vehicle is following road rules** (stopping at red lights, yielding to pedestrians, staying in correct lanes).

    Scene Description: {temporal_answer}


    **Return the output formatted as a json dictionary as follows:**
    Example: 
    
    "street_environment": "The ego vehicle is on a city street near an intersection with visible crosswalks and traffic signals.",
    "ego_positioning": "The ego vehicle is in the rightmost lane, approaching a traffic light. A pedestrian is crossing ahead.",
    "road_rule_following": "The ego vehicle is correctly positioned and following the speed limit but may need to yield to a pedestrian.",
    "action_recommendation": "The vehicle should prepare to stop at the red light and yield to the pedestrian."
    

    '''
    
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
    response = model.generate_content(contents=[prompt6, last_frame])

    json_string = re.sub(r"```json|```", "", response.text).strip()
    dic = json.loads(json_string)
    
    reasoning_enc = " ".join(str(value) for value in dic.values())
    
    # print("\n\n Reasoning: ", reasoning_enc)
    # print("\n\n temporal: ", temporal_answer)
    all_information = f"{temporal_answer} {reasoning_enc}"
    return all_information
    
# Part 6: Final output 
def output_fun(description, mcq_question, mcq_answers):
    prompt7 = f'''
    Ego is an autonomous vehicle navigating the streets, \
                            following road rules, avoiding pedestrians and objects, and responding to traffic \
                            signs, speed limits, and parking restrictions. It must also be aware of nearby vehicles.
    
    You are provided with a description of a video taken from ago as well as reasoned information about other objects in relation to the car, a multiple choice question and associated answers. Select the most appropriate answer. 
    
    Description: {description}
    
    Multiple Choice Question: {mcq_question}
    
    Multiple Choice Answers: {mcq_answers}
    
    **Return the output formatted as a JSON dictionary as follows:**
    
    "best_answer": "A",
    "reasoning": "Based on the scene description, the ego vehicle is approaching a traffic light with pedestrians in the crosswalk. The correct answer is A, as it aligns with the vehicle needing to yield."
    
    '''
    
    response = call_openai_gpt(prompt7)
    # print("response", response)
    
    json_string = re.sub(r"```json|```", "", response).strip()
    dic = json.loads(json_string)
    
    return dic["best_answer"]
    
  
def read_csv(input_file):
    content = []
    first = True
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if first: 
                first = False
                pass
            else:
                components = row[1].split('A.')
                components[1] = 'A.'+components[1] 
                content.append(components)
     
    return content

def main():
    
    images = ['/Users/nanaki/Desktop/10-.png','/Users/nanaki/Desktop/11-.png', '/Users/nanaki/Desktop/12-.png', '/Users/nanaki/Desktop/13-.png', '/Users/nanaki/Desktop/14-.png', '/Users/nanaki/Desktop/15-.png', '/Users/nanaki/Desktop/16-.png', '/Users/nanaki/Desktop/17-.png','/Users/nanaki/Desktop/18-.png', '/Users/nanaki/Desktop/19-.png', '/Users/nanaki/Desktop/20-.png']
    input_csv = "/Users/nanaki/treehacks25/tesla-real-world-video-q-a/questions.csv"  # Replace with your actual file name
    
    content_csv = read_csv(input_csv)

    for i in range(16,21):
        video_path = f'tesla-real-world-video-q-a/videos/videos/000{i}.mp4'
        associated_q = content_csv[i-1]
        mcq_question = associated_q[0]
        mcq_answers = associated_q[1]
        
        image_path = f'/Users/nanaki/Desktop/{i}-.png'
                
        
#         mcq_question = "What is the white car most likely to do at the stop sign?"
#         mcq_answers = "A. turn right. B. full stop. C. proceed straight. D. turn left."
#         image_path = "/Users/nanaki/Desktop/40.png"
#         video_path = 'tesla-real-world-video-q-a/videos/videos/00040.mp4'
        
    
#     # another 
#     # mcq_question = "What is the status of the traffic light?"
#     # mcq_answers = "A. Solid green. B. Blinking green. C. Solid red. D. Blinking red."
#     # image_path = "/Users/nanaki/Desktop/11.png"
#     # video_path = 'tesla-real-world-video-q-a/videos/videos/00011.mp4'
    
        
        
        start_time = time.time()
        

        # mcq_question = "Why is it necessary for the car to slow down?"
        # mcq_answers = "A. For another vehicle. B. For a pedestrian. C. For an animal. D. For a traffic light."

        # Step 1: Extract Features
        key_features = extract_key_features(mcq_question, mcq_answers)
        # print("\nKey Features Identified:\n", key_features)

        # Step 2: Generate Perception Questions
        perception_questions = generate_perception_questions(key_features)
        # print("\nGenerated Perception Questions:\n", perception_questions)
        
        # Convert perceptions questions to list
        perception_data = extract_questions(json.loads(perception_questions))
        # print(perception_data)
        
        # Meta CoT
        # image_path = "/Users/nanaki/Desktop/11.png"
        perception_questions_valid = validate_features_with_image(image_path, perception_questions, mcq_question)
        # print("\n, new", perception_questions_valid)
        
        # print(len(perception_data))
        # print(len(perception_questions_valid))
        
        # Step 4 - works 
        # video_path = 'tesla-real-world-video-q-a/videos/videos/00018.mp4'
        saved_frames, last_frame = create_video_subsamples(video_path, perception_questions, frame_interval=35)
        
        # Step 5
        collated_t = collate_temporaly(saved_frames)
        # print("collated tempor: ", collated_t)
        
        # Step 6 
        reasoning_enc = reasoning_encoding(last_frame, collated_t)
        
        # Step 7
        
        final_ans = output_fun(reasoning_enc, mcq_question, mcq_answers)
        
        print("Frame:", video_path)
        print("\nFinal: ", final_ans)
        
        end_time = time.time() 
        execution_time = end_time - start_time  # Calculate execution time
        print(f"Execution time: {execution_time:.4f} seconds")
        
        
    
    
if __name__ == "__main__":
    
    # video_paths = os.listdir('tesla-real-world-video-q-a/videos/videos')
    # video_path = os.path.join('tesla-real-world-video-q-a/videos/videos', video_paths[0])
    
    
    main()
    
