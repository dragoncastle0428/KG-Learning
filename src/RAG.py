import json
import os
from dotenv import load_dotenv
from google.genai import Client
from datasets import Dataset, load_from_disk # 🌟 Hugging Face Datasets 추가

# --- 1. 설정 ---
load_dotenv()
client = Client(api_key=os.getenv("GEMINI_API_KEY2"))

# 일부 핵심 관계명 매핑 (Baseline 가독성 향상)
RELATION_MAP = {
    "P159": "headquarters location", "P17": "country", 
    "P131": "located in", "P150": "contains", 
    "P576": "dissolved date", "P355": "subsidiary", 
    "P749": "parent organization"
}

def get_entity(vertex_set, idx):
    """vertexSet에서 인덱스 에러 방지용 안전한 엔티티 추출기"""
    try:
        return vertex_set[idx][0]['name']
    except (IndexError, KeyError):
        return "[Unknown Entity]"

def run_isolated_rag_test(orig_path, enriched_file, target_index=0):
    # --- 2. [A] 원본 Re-DocRED 데이터 로드 (.arrow 지원) ---
    print("⏳ 데이터를 불러오는 중입니다...")
    try:
        # 1. 만약 경로가 폴더 형태(save_to_disk로 저장한 경우)라면:
        if os.path.isdir(orig_path):
            orig_dataset = load_from_disk(orig_path)
            # train 셋이라고 가정하고 가져옴
            if 'train' in orig_dataset:
                orig_doc = orig_dataset['train'][target_index]
            else:
                orig_doc = orig_dataset[target_index]
        # 2. 만약 경로가 정확히 .arrow 파일 하나를 가리킨다면:
        elif orig_path.endswith('.arrow'):
            orig_dataset = Dataset.from_file(orig_path)
            orig_doc = orig_dataset[target_index]
        else:
            raise ValueError("지원하지 않는 원본 데이터 형식입니다.")
            
    except Exception as e:
        print(f"❌ 원본 데이터를 불러오는 데 실패했습니다: {e}")
        return

    # --- 3. [B] 새로 구축한 HypeRelDocRED 데이터 로드 (.jsonl) ---
    with open(enriched_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        enriched_doc = json.loads(lines[target_index])
        
    print(f"📊 [테스트 문서]: {orig_doc.get('title', 'Unknown')}\n")

    # --- 4. 문맥 독립 생성 ---
    
    # [A] Baseline Context (오직 원본 .arrow 문서의 데이터만 사용)
    baseline_context = ""
    for lbl in orig_doc.get('labels', []):
        h = get_entity(orig_doc['vertexSet'], lbl['h'])
        t = get_entity(orig_doc['vertexSet'], lbl['t'])
        r = RELATION_MAP.get(lbl['r'], lbl['r'])
        baseline_context += f"- {h} ({r}) {t}\n"

    # [B] Hyper-Relational Context (새 문서 .jsonl 의 데이터만 사용)
    hyper_context = ""
    for h_lbl in enriched_doc.get('hyper_labels', []):
        if not h_lbl or 'fact_id' not in h_lbl: continue
        
        # 주어(Head) 매핑
        fact_id = h_lbl['fact_id']
        try:
            h_idx = enriched_doc['labels'][fact_id]['h']
            h = get_entity(enriched_doc['vertexSet'], h_idx)
        except Exception:
            h = "[Unknown]"

        rel = h_lbl.get('refined_relation', '')
        t = h_lbl.get('refined_tail', '')
        if t is None: t = "" 
        
        event = f"{h} {rel} {t}".strip()
        qualifiers = ", ".join([f"{q['qualifier_relation']}: {q['qualifier_value_text']}" for q in h_lbl.get('qualifiers', [])])
        
        if qualifiers:
            hyper_context += f"- Event: {event} | Context: ({qualifiers})\n"
        else:
            hyper_context += f"- Event: {event}\n"

    # --- 5. 복합 질의 및 LLM 평가 ---
    query = "2016년 1월에 Zest Air에게 구체적으로 어떤 일이 일어났으며, 그 결과 어떻게 되었나요?"

    prompt = f"""
    당신은 두 가지 형태의 지식 그래프 시스템의 성능을 비교하는 AI 평가관입니다.
    아래 질문에 대해 [Baseline Context]와 [Hyper-Relational Context]를 각각 독립적으로 참고하여 답변을 작성하고 분석하세요.

    질문: {query}

    [Baseline Context] (원본 Re-DocRED 정적 팩트)
    {baseline_context}

    [Hyper-Relational Context] (사건/시간 중심 하이퍼 릴레이셔널 팩트)
    {hyper_context}

    출력 형식:
    ---
    🔴 [원본 기반 답변]: (정보 부족 시 모른다고 할 것)
    🔵 [신규 기반 답변]: 
    ⚖️ [비교 분석]: (시간 및 주체/객체의 연결성에 초점)
    ---
    """

    print("🧠 모델이 완벽히 격리된 두 데이터를 비교 분석 중입니다...\n")
    
    response = client.models.generate_content(
        model='models/gemini-2.5-flash',
        contents=prompt
    )
    
    print(response.text)

if __name__ == "__main__":
    # 🌟 경로 수정 가이드:
    # 1. Hugging Face에서 save_to_disk()로 저장한 '폴더'인 경우: "./Re-DocRED_disk" (또는 내부의 train 폴더)
    # 2. 직접적인 arrow 파일인 경우: "./Re-DocRED_disk/data-00000-of-00001.arrow" (실제 파일명)
    
    ORIGINAL_PATH = "./Dataset/Re-DocRED/train" # <-- 여기에 맞게 수정해주세요!
    ENRICHED_FILE = "./Dataset/HypeRelDocRED/train_enriched.jsonl" 
    
    run_isolated_rag_test(ORIGINAL_PATH, ENRICHED_FILE, target_index=0)