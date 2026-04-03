from datasets import load_dataset

# dataset = load_dataset("declare-lab/HyperRED")

# # 지정한 경로에 데이터셋 저장
# dataset.save_to_disk("./Dataset/HyperRED")

# dataset = load_dataset("tonytan48/Re-DocRED")

# # 지정한 경로에 데이터셋 저장
# dataset.save_to_disk("./Dataset/Re-DocRED")

from datasets import load_from_disk
from pprint import pprint

# 1. 파일들이 모여있는 폴더 경로를 지정하여 데이터셋 로드
# (주의: .arrow 파일 경로가 아니라, 그 파일이 들어있는 '폴더' 경로를 넣어야 합니다)
dataset_path = "./Dataset/Re-DocRED/"

print("데이터셋 불러오는 중...")
local_dataset = load_from_disk(dataset_path)

# 2. 데이터셋 구조 확인
print("\n=== 📊 데이터셋 구조 ===")
print(local_dataset)

# 3. 첫 번째 데이터(Row) 확인
print("\n=== 📝 첫 번째 샘플 데이터 (Train Split) ===")
pprint(local_dataset['train'][0], sort_dicts=False)
