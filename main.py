from retrieval import retrieve
import json

with open("descriptions/test.json", "r", encoding="utf-8") as f:
    descriptions = json.load(f)

for description in descriptions:
    query = [description["positive"], description["negative"]]
    picture_id = retrieve(query)
    description["retrieved_picture_id"] = picture_id

with open("descriptions/picture_id.json", "r", encoding="utf-8") as f:
    picture_id = json.load(f)

# Add the path to each description based on the retrieved picture_id
for description in descriptions:
    retrieved_id = description.get("retrieved_picture_id")
    if retrieved_id in picture_id:
        description["path"] = picture_id[retrieved_id]
    else:
        description["path"] = None  # Handle case where ID not found


with open("descriptions/test_results.json", "w", encoding="utf-8") as f:
    json.dump(descriptions, f)
    