import os
import json
import time
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types
from datasets import load_from_disk
from pydantic import BaseModel, Field
from typing import List

# 1. Wikidata P-ID 매핑 딕셔너리 (전수 검증 완료 버전)
RELATION_MAP = {
    # 🏢 [조직/기업 관련 (ORG)]
    "P571": "inception / founded in",         # 설립일/창립일
    "P112": "founded by",                     # 창립자
    "P159": "headquarters location",          # 본사/중심지 위치
    "P127": "owned by",                       # 소유주 (모회사/지주회사)
    "P749": "parent organization",            # 상위 조직
    "P355": "subsidiary",                     # 자회사/하위 조직
    "P155": "follows / preceded by",          # 이전 이름/전신 
    "P156": "followed by",                    # 이후 이름/후신
    "P1365": "replaces",                      # ~를 대체함/인수함
    "P1366": "replaced by",                   # ~로 대체됨/인수합병됨
    "P1416": "affiliation / strategic alliance", # 💡수정됨 (제휴/협력/관계사)
    "P452": "industry",                       # 소속 산업군
    "P7888": "merged into",
    "P121": "operated",

    # 👤 [인물 관련 (PER)]
    "P569": "date of birth",                  # 출생일
    "P570": "date of death",                  # 사망일
    "P19": "place of birth",                  # 출생지
    "P20": "place of death",                  # 사망지
    "P106": "occupation",                     # 직업
    "P108": "employer",                       # 고용주/소속 직장
    "P69": "educated at",                     # 출신 학교/학력
    "P26": "spouse",                          # 배우자
    "P22": "father",                          # 아버지
    "P25": "mother",                          # 어머니

    # 🌍 [지리/장소 관련 (LOC)]
    "P17": "country",                         # 소속 국가
    "P131": "located in",                     # 위치한 행정 구역 (도시, 주 등)
    "P36": "capital",                         # 수도
    "P47": "shares border with",              # 인접 지역/국경을 맞댐
    "P1992": "proximate points of interest", 

    # 🎬 [사건/창작물/기타 (TIME, EVENT, WORK)]
    "P585": "point in time",                  # 특정 사건이 일어난 시점
    "P577": "publication date",               # 출판/발표일
    "P50": "author",                          # 저자
    "P57": "director",                        # 감독
    "P175": "performer",                      # 연주자/출연자
    "P166": "award received",                 # 수상 내역
    "P1411": "nominated for",                 # 후보 지명
    
    "P1056": "service provided",              # 💡수정됨 (서비스/제품 제공)
    "P1444": "destination point",             # 💡수정됨 (여정/항공편의 목적지)
    "P3608": "regulated by"                   # 💡수정됨 (규제/관리 감독 기관)
}

# --- 1. 설정 및 클라이언트 초기화 ---
load_dotenv()
client = Client(api_key=os.getenv("GEMINI_API_KEY2"))

# --- 2. 데이터 구조 정의 (LLM은 무조건 String만 뱉도록 강제) ---
class Qualifier(BaseModel):
    qualifier: str = Field(description="e.g., WHEN, WHERE, WHY, HOW")
    qualifier_value: str = Field(description="Contextual detail (within 5 words)")

class Label(BaseModel):
    h: str = Field(description="The exact text of the Head Entity")
    h_type: str = Field(description="Type of Head (e.g., INS, LOC, PER, TIME, NUM)") # 추가됨
    r: str = Field(description="The exact Wikidata P-ID or Extracted Action/Relation")
    t: str = Field(description="The exact text of the Tail Entity")
    t_type: str = Field(description="Type of Tail (e.g., INS, LOC, PER, TIME, NUM)") # 추가됨
    qualifiers: List[Qualifier] = Field(default_factory=list, description="List of qualifiers")

class DocumentEnrichmentResult(BaseModel):
    labels: List[Label]

# --- 3. 헬퍼 함수 ---
def flatten_document(sents: list) -> str:
    return " ".join([" ".join(sent) for sent in sents])

def find_exact_pos_and_sent_id(sents: list, target_text: str) -> dict:
    """원문 sents 배열에서 특정 텍스트의 sent_id와 pos(토큰 좌표)를 정밀 추적합니다."""
    target_tokens = [t.lower() for t in target_text.split()]
    target_len = len(target_tokens)
    
    for sent_idx, sentence_tokens in enumerate(sents):
        sent_tokens_lower = [t.lower() for t in sentence_tokens]
        for i in range(len(sent_tokens_lower) - target_len + 1):
            if sent_tokens_lower[i : i + target_len] == target_tokens:
                return {"sent_id": sent_idx, "pos": [i, i + target_len]}
                
    return {"sent_id": -1, "pos": [-1, -1]} # 매칭 실패 시

def rebuild_docred_format(original_vertex_set: list, sents: list, llm_labels_text: list):
    """LLM이 판별한 타입을 보존하면서 새로운 구조로 재조립합니다."""
    new_vertex_set = []
    new_labels = []

    # 💡 수정 1: entity_name 뿐만 아니라 entity_type도 파라미터로 같이 받습니다.
    def get_or_create_vertex(entity_name: str, entity_type: str) -> int:
        entity_name_clean = entity_name.strip().lower()
        
        # 1. 이미 새로 만든 vertexSet에 있는지 확인 (중복 방지)
        for i, entity_group in enumerate(new_vertex_set):
            if entity_group[0]['name'].strip().lower() == entity_name_clean:
                return i

        # 2. 원본 검색 로직(CONCEPT 폴백) 완전히 삭제!
        # sents에서 좌표(pos, sent_id)만 추적합니다.
        location = find_exact_pos_and_sent_id(sents, entity_name)
        
        # 3. 새로운 노드 생성
        new_node = [{
            "name": entity_name.strip(),
            "type": entity_type,  # 💡 수정 2: LLM이 판별한 타입(ORG, LOC 등)을 그대로 박아 넣습니다!
            "sent_id": location["sent_id"],
            "pos": location["pos"]
        }]
        
        new_vertex_set.append(new_node)
        return len(new_vertex_set) - 1

    for lbl in llm_labels_text:
        # 💡 수정 3: 노드를 만들 때 LLM이 생성한 h_type과 t_type을 함께 넘겨줍니다.
        h_idx = get_or_create_vertex(lbl.h, lbl.h_type)
        t_idx = get_or_create_vertex(lbl.t, lbl.t_type)
        
        h_sent = new_vertex_set[h_idx][0]["sent_id"]
        t_sent = new_vertex_set[t_idx][0]["sent_id"]
        evidence_set = {h_sent, t_sent}
        evidence_set.discard(-1) 
        
        new_labels.append({
            "h": h_idx,
            "r": lbl.r,
            "t": t_idx,
            "evidence": sorted(list(evidence_set)),
            "qualifiers": [q.model_dump() for q in lbl.qualifiers]
        })

    return new_vertex_set, new_labels

# --- 4. LLM API 호출 함수 ---
def enrich_document_batch(doc_text: str, doc_title: str):
    relation_menu_str = "\n".join([f"       - {pid}: {desc}" for pid, desc in RELATION_MAP.items()])

    prompt = f"""
    You are an Expert Data reconstructor for Knowledge Graphs. 
    Analyze the document and extract core relationships.
    
    [CRITICAL INSTRUCTIONS]
    1. COREFERENCE RESOLUTION (MANDATORY): You MUST NOT use pronouns or generic nouns for entities. You MUST resolve them to their specific proper noun.
    2. CONCISE ENTITIES: Entities must be short and concise (1 to 4 words maximum). Remove unnecessary adjectives and determiners except for specific name of departments, organizations, or locations.
    3. ENTITY EXTRACTION Priority: {doc_title} is the title of the document. Consider it as the central entity if possible.
    4. Use conditional details (dates, locations, reasons) to 'qualifiers'. (qualifier: WHEN, WHERE, WHY, HOW / qualifier_value: concise contextual detail within 5 words)
    5. 'r' MUST be a specific action including services or relation, not a vague one. Head and Tail MUST be exact entities from the text that can be explained.
    6. 'r' MUST be a Wikidata P-ID which is listed in the menu below. If the relation is not explicitly stated as a P-ID in the text, you MUST infer the MOST appropriate P-ID that is listed.
    7. ENTITY TYPING: For every extracted 'h' and 't', you MUST assign one of the following exact types in 'h_type' and 't_type':
       - ORG (Organization / Company / Institution / Department / Government)
       - LOC (Location / Facility)
       - PER (Person)
       - TIME (Date / Year / Duration)
       - NUM (Number / Amount)
    8. Entity Unification (Strict): If the same entity appears multiple times in the document with slight variations (e.g. changing word order), you MUST unify them into a single canonical form and use that consistently across all labels.)
        Apply these general normalization rules:
       - Pattern A (Acronyms): Map abbreviations to full names (e.g., "WHO" -> "World Health Organization").
       - Pattern B (Word Order): Map swapped word orders to a standard name (e.g., "America Bank" -> "Bank of America").
       - Pattern C (Partial Names): Map shortened names or generic references to the full proper noun (e.g., "the airline" -> "Specific Airline Airlines").
    9. Do not extract entities that is describing property of another entity. Instance of, part of and other similar descriptive relations should be captured as qualifiers or not included at all.
    10. AVOID ISOLATED ISLANDS: Every extracted relationship MUST connect to the central subjects of the text. Do not create disconnected node pairs that float independently.

    P-ID Relation Menu:
    {relation_menu_str}

    [Document]
    {doc_text}
    """
    try:
        response = client.models.generate_content(
            model='models/gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=DocumentEnrichmentResult,
                temperature=0,
            ),
        )
        return response.parsed
    except Exception as e:
        print(f"  ⚠️ API 에러: {e}")
        return None

# --- 5. 메인 처리 로직 ---
def process_split(split_name: str, dataset_path: str, output_base_dir: str):
    dataset = load_from_disk(dataset_path)[split_name]
    os.makedirs(os.path.join(output_base_dir, split_name), exist_ok=True)
    output_file = os.path.join(output_base_dir, split_name, f"{split_name}_enriched_v2.jsonl")
    
    start_idx = sum(1 for _ in open(output_file, 'r')) if os.path.exists(output_file) else 0
    print(f"🚀 [{split_name}] {start_idx}번 문서부터 시작합니다.")
    start_idx = 15

    for i in range(start_idx, start_idx + 1): # 테스트 시 range(start_idx, start_idx + 1) 로 변경하세요
        sample = dataset[i]
        doc_text = flatten_document(sample['sents'])
        
        print(f"[{split_name}] [{i+1}/{len(dataset)}] 처리 중...", end=" ", flush=True)
        
        # 1. LLM 추출 (텍스트만 뱉음)
        batch_res = enrich_document_batch(doc_text, sample['title'])
        
        if batch_res:
            # 2. 파이썬 후처리 (텍스트 -> 좌표 매핑 및 vertexSet/labels 조립)
            new_vset, new_lbls = rebuild_docred_format(sample['vertexSet'], sample['sents'], batch_res.labels)
            
            # 3. 데이터 덮어쓰기
            sample['vertexSet'] = new_vset
            sample['labels'] = new_lbls
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            print("✅")
        else:
            print("❌ (API 실패 - 스킵됨)")
        
        time.sleep(4.5)

# --- 6. 실행 ---
if __name__ == "__main__":
    SOURCE_PATH = "./Dataset/Re-DocRED"
    OUTPUT_PATH = "./Dataset/HypeRelDocRED"
    
    for split in ["train"]:
        process_split(split, SOURCE_PATH, OUTPUT_PATH)