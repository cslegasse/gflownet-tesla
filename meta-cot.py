import google.generativeai as genai
import os
import csv
import re

def read_csv(input_file):
    content = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            content.append(row[1])
    return content

# Example usage
input_csv = "/Users/nanaki/treehacks25/tesla-real-world-video-q-a/questions.csv"  # Replace with your actual file name
content_csv = read_csv(input_csv)

video_paths = os.listdir('tesla-real-world-video-q-a/videos/videos')

for i in range(len(video_paths)):
    video_path = os.path.join('tesla-real-world-video-q-a/videos/videos', video_paths[i])
    ind = int(video_paths[i][-6:-4])
    associated_q = content_csv[ind]
    
    print(f"Uploading file...")
    video_file = genai.upload_file(path=video_path)
    print(f"Completed upload: {video_file.uri}")

    import time

    while video_file.state.name == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
      raise ValueError(video_file.state.name)
    print(f'Video processing complete: ' + video_file.uri)

    # Create the prompt.
    prompt = associated_q
    
    # Set the model to Gemini Flash.
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    # Make the LLM request.
    print("Making LLM inference request...")
    response = model.generate_content([prompt, video_file],
                                      request_options={"timeout": 600})
    print("\n", f"Data point: {ind} - ", prompt)
    print("\n", response.text, "\n",)
    
    

prompt3 = "Did ego execute a legal right turn maneuver at the intersection, considering lane occupancy, road markings, traffic laws, construction obstructions, and the feasibility of completing the turn safely? Argue why it cannot be the other options\
             A. It's legal as the lane is empty. \
                B. It's illegal as the right turn lane is blocked by construction. \
                C. It's illegal as ego was cutting in other vehicles that were waiting. \
                D. It's legal but the lane ahead is way too narrow for ego to pass."
                
                
prompt2 = '''I need your help in creating a new question that queries on more semantic details in order to extract more vital information from a video clip. The video clip is taken from the perspective of 'ego', the dashboard of an autonomous vehicle. 

                Here is a multiple choice question: 
                Was ego doing a legal maneuver if its goal is to turn right at the intersection?

                Here are the options for it:
                A. It's legal as the lane is empty. 
                B. It's illegal as the right turn lane is blocked by construction. 
                C. It's illegal as ego was cutting in other vehicles that were waiting. 
                D. It's legal but the lane ahead is way too narrow for ego to pass.

                Briefly, using only the information available from the question and answers, list out all the most important features that I should pay attention to in the video clip.

                Here is an example answer:
                Example multiple choice question:
                What is the reason ego changed lanes to the left? 

                Example answers:
                A.  Left lane has better views. 
                B. Current lane is exit only. 
                C. Current lane has a lower speed limit. 
                D. Current lane is blocked.

                Example response from GPT:
                Important features to evaluate:
                - Other vehicles on the road and in the path in front of ego
                - Signage for construction/blocking on the road
                - Signage for speed 
                - Signage for vehicle restrictions in certain lanes
                - Road type that ego is driving on 
                
                Only output important features to evaluate in bullet point format.
                '''


