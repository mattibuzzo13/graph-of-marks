import json
import os

# Define paths
data_dir = 'gqa'
ann_file = os.path.join(data_dir, 'annotations', 'train_all_questions.json')
img_dir = os.path.join(data_dir, 'images')

# Load annotations
with open(ann_file, 'r') as f:
    gqa_data = json.load(f)

# Example: Accessing the first question
first_question_id = list(gqa_data.keys())[0]
first_question = gqa_data[first_question_id]
image_id = first_question['imageId']
question_text = first_question['question']
answer = first_question['answer']

# Load the corresponding image
image_filename = f'{image_id}.jpg'
image_path = os.path.join(img_dir, image_filename)

print(f"Image Path: {image_path}")
print(f"Question: {question_text}")
print(f"Answer: {answer}")
