from datasets import load_dataset

dataset = load_dataset("declare-lab/HyperRED")

# 지정한 경로에 데이터셋 저장
dataset.save_to_disk("./Dataset/HyperRED")
