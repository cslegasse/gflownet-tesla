from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if __name__ =="__main__":
  client = genai.Client(api_key=GOOGLE_API_KEY)
  files = client.files.list()
  for file in files:
    print(file.name)
    client.files.delete(name=file.name)
    
  # flush gemini_cache.csv
  with open("gemini_cache.csv", "w") as f:
    f.write("")
  print("Gemini cache cleared.")