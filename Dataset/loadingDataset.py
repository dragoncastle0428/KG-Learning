from datasets import load_dataset

# dataset = load_dataset("declare-lab/HyperRED")

# # 지정한 경로에 데이터셋 저장
# dataset.save_to_disk("./Dataset/HyperRED")

# dataset = load_dataset("tonytan48/Re-DocRED")

# # 지정한 경로에 데이터셋 저장
# dataset.save_to_disk("./Dataset/Re-DocRED")

# from datasets import load_from_disk
# from pprint import pprint

# # 1. 파일들이 모여있는 폴더 경로를 지정하여 데이터셋 로드
# # (주의: .arrow 파일 경로가 아니라, 그 파일이 들어있는 '폴더' 경로를 넣어야 합니다)
# dataset_path = "./Dataset/Re-DocRED/"

# print("데이터셋 불러오는 중...")
# local_dataset = load_from_disk(dataset_path)

# # 2. 데이터셋 구조 확인
# print("\n=== 📊 데이터셋 구조 ===")
# print(local_dataset)

# # 3. 첫 번째 데이터(Row) 확인
# print("\n=== 📝 첫 번째 샘플 데이터 (Train Split) ===")
# pprint(local_dataset['train'][0], sort_dicts=False)

import os
from datasets import Dataset, load_from_disk

def convert_arrow_to_json(input_path, output_jsonl_path):
    print(f"⏳ [{input_path}] 데이터를 불러오는 중...")
    
    try:
        # 1. 데이터 로드 (폴더 형태 vs arrow 단일 파일 형태)
        if os.path.isdir(input_path):
            dataset = load_from_disk(input_path)
            # 만약 train/dev/test 스플릿이 나뉘어 있다면 train만 선택
            if 'train' in dataset:
                dataset = dataset['train']
        elif input_path.endswith('.arrow'):
            dataset = Dataset.from_file(input_path)
        else:
            print("❌ 지원하지 않는 경로/파일 형식입니다.")
            return

        print(f"✅ 데이터 로드 성공! 총 {len(dataset)}개의 문서를 변환합니다.")

        # 2. JSONL 파일로 내보내기 (to_json 메서드 사용)
        # force_ascii=False를 주어야 텍스트가 깨지지 않고 원형 그대로 보존됩니다.
        dataset.to_json(output_jsonl_path, force_ascii=False, lines=True)
        
        print(f"🎉 변환 완료! [{output_jsonl_path}] 파일이 생성되었습니다.")

    except Exception as e:
        print(f"❌ 변환 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    # 🌟 여기에 사용자님의 원본 데이터 경로와 저장할 파일 이름을 적어주세요!
    INPUT_PATH = "./Dataset/HyperRED/train" # 예: 폴더 경로 또는 .arrow 파일 경로
    OUTPUT_PATH = "hyperred_original.jsonl"
    
    convert_arrow_to_json(INPUT_PATH, OUTPUT_PATH)
