import os
import json
import time
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types  # 구조화된 출력을 위해 필요
from datasets import load_from_disk
from pydantic import BaseModel, Field
from typing import List

# --- 1. 설정 및 클라이언트 초기화 ---
load_dotenv()
client = Client(api_key=os.getenv("GEMINI_API_KEY2"))

RELATION_MAP = {
    "P159": "headquarters location", "P17": "country",
    "P576": "dissolved date", "P355": "subsidiary", "P749": "parent organization"
}

# --- 2. 데이터 구조 정의 (Pydantic - 필드 완벽 추가) ---
class Qualifier(BaseModel):
    qualifier: str = Field(description="e.g., WHEN, WHERE, WHY, HOW")
    qualifier_value: str = Field(description="Contextual detail (within 5 words)")

class Label(BaseModel):
    evidence: List[int] = Field(description="The sentence indices in the document that support this fact")
    h: int = Field(description="The index of the Canonical Head Entity")
    r: str = Field(description="The exact Wikidata P-ID or Extracted Action/Relation")
    t: int = Field(description="The index of the Canonical Tail Entity OR Extracted Concept")
    qualifiers: List[Qualifier] = Field(default_factory=list, description="List of qualifiers for this relation")

class DocumentEnrichmentResult(BaseModel):
    labels: List[Label]

# --- 3. 헬퍼 함수 ---
def flatten_document(sents: list) -> str:
    return " ".join([" ".join(sent) for sent in sents])

def get_entity_name(vertex_set: list, entity_idx: int) -> str:
    return vertex_set[entity_idx][0]['name']

# --- 4. 핵심 추출 함수 ---
def enrich_document_batch(doc_text: str, facts_to_analyze: list):
    if not facts_to_analyze: return None

    facts_str = "\n".join([f"[evidence: {f['evidence']}] {f['h']} --({f['r']})--> {f['t']}" for f in facts_to_analyze])
    
    prompt = f"""
    You are an Expert Data reconstructor for Knowledge Graphs. Your task is to analyze the provided document and the associated base facts, then output a refined set of relationships in a structured JSON format.

    [CRITICAL PIPELINE INSTRUCTIONS: FOLLOW IN EXACT ORDER]

    **STEP 1: Global Entity Unification (The Most Important Step)**
    - Scan the entire text. Identify the primary subjects/entities.
    - If multiple names refer to the exact same entity (e.g., full names, acronyms, shortened names, pronouns like "it" or "the company"), you MUST mentally map ALL of them to a SINGLE, longest, most descriptive Canonical Name.
    - You must apply this mapping to BOTH the Head and Tail of every relationship.

    **STEP 2: Event and Concept Extraction**
    - Map "What happened", not just static states.
    - Consider all nouns as an entity, but only if they are involved in an importantaction or event.
    - If the text describes a specific activity, service, legal action, or physical incident, that Action/Concept MUST become a distinct node (Tail) connected to the main Entity (Head).
    - Ensure all extracted events connect back to the Canonical Entities, creating a cohesive "Hub and Spoke" topology rather than disconnected islands.

    **STEP 3: Aggregation and Purging**
    - If multiple facts state the exact same relationship between the exact same Canonical Head and Canonical Tail, group them into a single JSON object.
    - Do NOT output dates, times, or raw numbers as standalone `refined_tail` nodes. Move them to the `qualifiers` array.
    - If there is a date, time, or number in the vertexSet, remove it from the vertexSet and add it as a qualifier to the relevant relationship.
    - If the dates, times, or numbers are critical to multiple relationships, they should be included as entities (not in qualifier).

    [MANDATORY JSON STRUCTURE]
    You MUST output a single JSON object containing exactly ONE key named "labels". 
    DO NOT generate any other root-level keys. 
    Qualifier values must be short and concise. (within 5 words)
    The "labels" array must contain all facts and qualifiers (if exists) formatted exactly like this:
    [
      {{
        "evidence": <Array of integers: The sentence indices in the document that support this fact>,
        "h": <Integer: The index of the Canonical Name determined in STEP 1>,
        "r": <String: The exact Wikidata P-ID>,
        "t": <Integer: The index of the Canonical Name OR Extracted Concept>,
        "qualifiers": [ 
          {{
            "qualifier": <String: e.g., WHEN, WHERE, WHY, HOW>,
            "qualifier_value": <String: Contextual detail>
          }}
        ]
      }}
    ]
   

    [Document]
    {doc_text}

    [Base Facts to Restructure]
    {facts_str}

    Output the final JSON array strictly adhering to the pipeline.
    """

    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=DocumentEnrichmentResult,
                temperature=0.1,
            ),
        )
        return response.parsed
    except Exception as e:
        print(f"  ⚠️ API 에러: {e}")
        return None

# --- 5. 분할(Split)별 처리 및 이어하기 로직 ---
def process_split(split_name: str, dataset_path: str, output_base_dir: str):
    dataset = load_from_disk(dataset_path)[split_name]
    
    split_output_dir = os.path.join(output_base_dir, split_name)
    os.makedirs(split_output_dir, exist_ok=True)
    
    output_file = os.path.join(split_output_dir, f"{split_name}_enriched_v2.jsonl")
    
    # 이어하기 체크: 기존 파일의 줄 수 확인
    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"📦 [{split_name}] {start_idx}번 문서부터 작업을 재개합니다.")
    else:
        print(f"🚀 [{split_name}] 처음부터 처리를 시작합니다.")

    for i in range(start_idx, start_idx + 1):
        sample = dataset[i]
        doc_text = flatten_document(sample['sents'])
        
        facts_batch = []
        for idx, lbl in enumerate(sample['labels']):
            facts_batch.append({
                "evidence": idx,
                "h": get_entity_name(sample['vertexSet'], lbl['h']),
                "r": RELATION_MAP.get(lbl['r'], lbl['r']),
                "t": get_entity_name(sample['vertexSet'], lbl['t'])
            })

        print(f"[{split_name}] [{i+1}/{len(dataset)}] 처리 중...", end=" ", flush=True)
        
        # 성공할 때까지 혹은 에러 발생 시 대기 후 처리
        batch_res = enrich_document_batch(doc_text, facts_batch)
        
        if batch_res:
            # 💡 수정됨: hyper_labels 대신 기존 labels 구조를 대체(또는 병합)
            # Pydantic 모델(DocumentEnrichmentResult) 안에 있는 labels 리스트를 dict로 변환
            sample['labels'] = [lbl.model_dump() for lbl in batch_res.labels]
            
            # 한 줄씩 추가 저장
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print("✅")
        else:
            print("❌ (실패 - 스킵됨)")
        
        # 무료 API 안정성을 위해 4~5초 대기
        time.sleep(4.5)

# --- 6. 메인 실행 루프 ---
if __name__ == "__main__":
    SOURCE_PATH = "./Dataset/Re-DocRED"  # 기존 데이터셋 경로
    OUTPUT_PATH = "./Dataset/HypeRelDocRED"   # 결과 저장 경로
    
    for split in ["train"]:
        process_split(split, SOURCE_PATH, OUTPUT_PATH)


# # --- 테스트 실행 (방법 2: google-genai 최신 SDK 버전) ---
# if __name__ == "__main__":
#     # 1. 사용자님이 제공해주신 1개짜리 샘플 데이터
#     test_dataset = [{
#         'sents': [['Zest', 'Airways', ',', 'Inc.', 'operated', 'as', 'AirAsia', 'Zest', '(', 'formerly', 'Asian', 'Spirit', 'and', 'Zest', 'Air', ')', ',', 'was', 'a', 'low', '-', 'cost', 'airline', 'based', 'at', 'the', 'Ninoy', 'Aquino', 'International', 'Airport', 'in', 'Pasay', 'City', ',', 'Metro', 'Manila', 'in', 'the', 'Philippines', '.'], ['It', 'operated', 'scheduled', 'domestic', 'and', 'international', 'tourist', 'services', ',', 'mainly', 'feeder', 'services', 'linking', 'Manila', 'and', 'Cebu', 'with', '24', 'domestic', 'destinations', 'in', 'support', 'of', 'the', 'trunk', 'route', 'operations', 'of', 'other', 'airlines', '.'], ['In', '2013', ',', 'the', 'airline', 'became', 'an', 'affiliate', 'of', 'Philippines', 'AirAsia', 'operating', 'their', 'brand', 'separately', '.'], ['Its', 'main', 'base', 'was', 'Ninoy', 'Aquino', 'International', 'Airport', ',', 'Manila', '.'], ['The', 'airline', 'was', 'founded', 'as', 'Asian', 'Spirit', ',', 'the', 'first', 'airline', 'in', 'the', 'Philippines', 'to', 'be', 'run', 'as', 'a', 'cooperative', '.'], ['On', 'August', '16', ',', '2013', ',', 'the', 'Civil', 'Aviation', 'Authority', 'of', 'the', 'Philippines', '(', 'CAAP', ')', ',', 'the', 'regulating', 'body', 'of', 'the', 'Government', 'of', 'the', 'Republic', 'of', 'the', 'Philippines', 'for', 'civil', 'aviation', ',', 'suspended', 'Zest', 'Air', 'flights', 'until', 'further', 'notice', 'because', 'of', 'safety', 'issues', '.'], ['Less', 'than', 'a', 'year', 'after', 'AirAsia', 'and', 'Zest', 'Air', "'s", 'strategic', 'alliance', ',', 'the', 'airline', 'has', 'been', 'rebranded', 'as', 'AirAsia', 'Zest', '.'], ['The', 'airline', 'was', 'merged', 'into', 'AirAsia', 'Philippines', 'in', 'January', '2016', '.']],
#         'vertexSet': [[{'name': 'Zest Airways, Inc.'}], [{'name': 'Asian Spirit and Zest Air'}], [{'name': 'AirAsia Zest'}], [{'name': 'AirAsia Zest'}], [{'name': 'Ninoy Aquino International Airport'}], [{'name': 'Pasay City'}], [{'name': 'Metro Manila'}], [{'name': 'Philippines'}], [{'name': 'Manila'}], [{'name': 'Cebu'}], [{'name': '24'}], [{'name': '2013'}], [{'name': 'Philippines AirAsia'}], [{'name': 'Asian Spirit'}], [{'name': 'Civil Aviation Authority of the Philippines'}], [{'name': 'Zest Air'}], [{'name': 'a year'}], [{'name': 'AirAsia'}], [{'name': 'AirAsia Philippines'}], [{'name': 'January 2016'}]],
#         'title': 'AirAsia Zest',
#         'labels': [{'h': 0, 'r': 'P159', 't': 4}, {'h': 15, 'r': 'P576', 't': 19}]
#     }]
    
#     # 2. 결과 저장 경로 설정
#     # (주의: Dataset 폴더가 미리 생성되어 있어야 합니다)
#     OUT_PATH = "./Dataset/hyper_docred_v2_test.jsonl"
    
#     # 만약 기존에 테스트하던 파일이 있다면 삭제 (테스트 결과 중복 방지)
#     if os.path.exists(OUT_PATH):
#         os.remove(OUT_PATH)
#         print(f"🧹 기존 테스트 파일({OUT_PATH})을 삭제했습니다.")

#     print(f"🚀 테스트 샘플 1건 처리를 시작합니다...")
    
#     # 앞서 정의한 run_pipeline 함수 실행
#     # (이미 로딩된 test_dataset을 인자로 전달할 수 있도록 함수가 설계되어야 합니다)
#     # 만약 run_pipeline이 경로를 받는 구조라면 아래처럼 직접 호출하세요:
    
#     try:
#         # run_pipeline 내부에 파일 쓰기 로직이 포함되어 있으므로 그대로 호출
#         # (단, run_pipeline이 load_from_disk를 사용하는 경우 아래 리스트 직접 처리 루프로 대체)
        
#         for sample in test_dataset:
#             doc_text = flatten_document(sample['sents'])
            
#             facts_batch = []
#             for idx, lbl in enumerate(sample['labels']):
#                 facts_batch.append({
#                     "id": idx,
#                     "head": get_entity_name(sample['vertexSet'], lbl['h']),
#                     "relation": RELATION_MAP.get(lbl['r'], lbl['r']),
#                     "tail": get_entity_name(sample['vertexSet'], lbl['t'])
#                 })

#             print(f"🔍 '{sample['title']}' 분석 중 (팩트 {len(facts_batch)}개)...")
            
#             # API 호출
#             batch_res = enrich_document_batch(doc_text, facts_batch)
            
#             # 결과 병합
#             enriched_labels = sample['labels'].copy()
#             if batch_res:
#                 for res in batch_res.results:
#                     if res.fact_id < len(enriched_labels):
#                         # Pydantic 객체를 딕셔너리로 변환하여 저장
#                         enriched_labels[res.fact_id]['qualifiers'] = [q.model_dump() for q in res.qualifiers]
#                         for q in res.qualifiers:
#                             print(f"   ✨ 발견: [{q.qualifier_relation}] -> {q.qualifier_value_text}")
            
#             sample['labels'] = enriched_labels
            
#             # 파일 저장
#             os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
#             with open(OUT_PATH, 'a', encoding='utf-8') as f:
#                 f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
#             print(f"✅ 결과가 {OUT_PATH}에 저장되었습니다.")

#     except Exception as e:
#         print(f"❌ 테스트 중 에러 발생: {e}")