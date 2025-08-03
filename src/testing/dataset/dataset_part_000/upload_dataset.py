from huggingface_hub import login 
import json
from datasets import load_dataset, Dataset, Features, Value, Audio

HF_TOKEN=""
login(HF_TOKEN)
DATASET_NAME = "danyw24/colombian-speech-0.3"

SPLIT_NAME = "train"
data =[]

with open("./train_dataset.jsonl", "r") as file:
    for line in file:
        lin = json.loads(line)
        data.append(lin)

dataset = Dataset.from_list(data)
print("Dataset:", dataset)
print("Dataset structure:", dataset.features)


dataset = dataset.rename_column("audio_path", "audio")
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

print(dataset)
print(dataset.features) 

print(f"\nPushing dataset to {DATASET_NAME}...")
dataset.push_to_hub(DATASET_NAME, split=SPLIT_NAME)
