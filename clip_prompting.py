main_prompt = ''' Your job is to convert a MCQ question and associated answers to only a set \
    of bullet points that incoporate semantic information from the question. I will give you examples below:

Here is an MCQ with associated answers: 

What type of building is the road sign indicating is ahead? 
A. Airport. 
B. Embassy. 
C. Restaurant. 
D. Hotel.

The bullet points can be restructured to incoporate all the information from
the question, so that the question can be removed. I.e.:

A. The road sign indicates an Airport ahead. 
B. The road sign indicates an Embassy ahead. 
C. The road sign indicates an Restaurant ahead. 
D. The road sign indicates an Hotel ahead.

Another example: 

MCQ with associated answers: 

What is the white car most likely to do at the stop sign? 

A. turn right. 
B. full stop. 
C. proceed straight. 
D. turn left.

Restructured to incorporate information from question: 

A. The white car is most likely to turn right at the stop sign. 
B. The white car is most likely to full stop at the stop sign. 
C. The white car is most likely to proceed straight at the stop sign. 
D. The white car is most likely to turn left at the stop sign.


Another example: 

MCQ with associated answers: 

Is it safe for ego to park right here? 

A. yes there is a parking lot here. 
B. Yes the shoulder is large enough. 
C. No there's no safe place to pullover. 
D. No there's a fire station.

Restructured to incorporate information from question: 

A. It is safe for ego to park here because there is a parking lot here. 
B. It is safe for ego to park here because the shoulder is large enough. 
C. It is not safe for ego to park here because there is no safe place to pullover. 
D. It is not safe for ego to park here because there is a fire station.

'''

prompt_user = '''Now here is an MCQ with answers: 
Where is the pedestrian with respect to ego? 

A. On the crosswalk closest to ego. 
B. On the crosswalk on the left. 
C. On the crosswalk on the right. 
D. On the crosswalk at the opposite side of the intersection.

Reformat the answers
'''

from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": main_prompt}, {"role": "user", "content": prompt_user}])

print(completion.choices[0].message.content)