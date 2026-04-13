import os
import json
import time
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types
from datasets import load_from_disk
from pydantic import BaseModel, Field
from typing import List
import re
import difflib

# 1. Wikidata P-ID 매핑 딕셔너리 (전수 검증 완료 버전)
RELATION_MAP = {
    "P571": "founded in", "P112": "founded by", "P159": "headquarters", "P127": "owned by", 
    "P749": "parent org", "P355": "subsidiary", "P155": "preceded by", "P156": "followed by", 
    "P1365": "replaces", "P1366": "replaced by", "P1416": "affiliation", "P452": "industry", 
    "P7888": "merged into", "P121": "operated", "P569": "date of birth", "P570": "date of death", 
    "P19": "place of birth", "P20": "place of death", "P106": "occupation", "P108": "employer", 
    "P69": "educated at", "P26": "spouse", "P22": "father", "P25": "mother", "P17": "country", 
    "P131": "located in", "P36": "capital", "P47": "shares border", "P1992": "proximate points", 
    "P585": "point in time", "P577": "publication date", "P50": "author", "P57": "director", 
    "P175": "performer", "P166": "award received", "P1411": "nominated for", 
    "P1056": "service provided", "P1444": "destination point", "P3608": "regulated by",
    "P6": "head of government",
    "P27": "country of citizenship",
    "P30": "continent",
    "P31": "instance of",
    "P35": "head of state",
    "P37": "official language",
    "P39": "position held",
    "P40": "child",
    "P54": "member of sports team",
    "P58": "screenwriter",
    "P86": "composer",
    "P102": "member of political party",
    "P118": "league",
    "P123": "publisher",
    "P136": "genre",
    "P137": "operator",
    "P140": "religion",
    "P150": "contains administrative territorial entity",
    "P161": "cast member",
    "P162": "producer",
    "P170": "creator",
    "P171": "parent taxon",
    "P172": "ethnic group",
    "P176": "manufacturer",
    "P178": "developer",
    "P179": "series",
    "P194": "legislative body",
    "P205": "basin country",
    "P206": "located in or next to body of water",
    "P241": "military branch",
    "P264": "record label",
    "P272": "production company",
    "P276": "location",
    "P279": "subclass of",
    "P361": "part of",
    "P364": "original language",
    "P400": "platform",
    "P403": "mouth of the watercourse",
    "P449": "original network",
    "P463": "member of",
    "P488": "chair",
    "P495": "country of origin",
    "P527": "has part",
    "P551": "residence",
    "P576": "dissolved or abolished date",
    "P580": "start time",
    "P582": "end time",
    "P607": "conflict",
    "P674": "characters",
    "P676": "lyrics by",
    "P706": "located on terrain feature",
    "P710": "participant",
    "P737": "influenced by",
    "P740": "location of formation",
    "P800": "notable work",
    "P807": "separated from",
    "P840": "narrative location",
    "P937": "work location",
    "P1001": "legal jurisdiction",
    "P1198": "unemployment rate",
    "P1336": "territory claimed by",
    "P1344": "participant in",
    "P1376": "capital of",
    "P1412": "languages spoken or written",
    "P1441": "present in work",
    "P3373": "sibling"
}

load_dotenv()
API_KEYS = [os.getenv(f"GEMINI_API_KEY{i}") for i in range(1, 31) if os.getenv(f"GEMINI_API_KEY{i}")]

current_key_idx = 0

def get_current_client():
    """현재 인덱스의 API 키로 클라이언트를 생성하여 반환합니다."""
    global current_key_idx
    return Client(api_key=API_KEYS[current_key_idx])

# --- 1. 설정 및 클라이언트 초기화 ---

#client = Client(api_key=os.getenv(API_KEYS[0]))

# --- 2. 데이터 구조 정의 (LLM은 무조건 String만 뱉도록 강제) ---
class Qualifier(BaseModel):
    qualifier: str = Field(description="e.g., WHEN, WHERE, WHY, HOW")
    qualifier_value: str = Field(description="Contextual detail (within 5 words)")

class Label(BaseModel):
    h: str = Field(description="The exact text of the Head Entity")
    h_type: str = Field(description="Type of Head (e.g., ORG, LOC, PER, TIME, NUM)") # 추가됨
    r: str = Field(description="The exact Wikidata P-ID")
    t: str = Field(description="The exact text of the Tail Entity")
    t_type: str = Field(description="Type of Tail (e.g., ORG, LOC, PER, TIME, NUM)") # 추가됨
    qualifiers: List[Qualifier] = Field(default_factory=list, description="List of qualifiers")

class DocumentEnrichmentResult(BaseModel):
    labels: List[Label]

# --- 3. 헬퍼 함수 ---
def flatten_document(sents: list) -> str:
    return " ".join([" ".join(sent) for sent in sents])

def clean_text(text: str) -> str:
    """특수기호를 모두 제거하고 소문자로 변환하여 알파벳/숫자만 남깁니다."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

def find_robust_pos_and_sent_id(sents: list, target_text: str) -> dict:
    """
    1단계: 완벽 일치 (Exact Match)
    2단계: 특수기호 제거 후 일치 (Clean Match)
    3단계: 유사도 기반 퍼지 매칭 (Fuzzy Match)
    """
    target_clean = clean_text(target_text)
    target_tokens = target_text.lower().split()
    target_len = len(target_tokens)
    
    if not target_clean:
        return {"sent_id": -1, "pos": [-1, -1]}

    # --- [1단계] Exact Match (기존 방식 - 가장 빠르고 정확함) ---
    for sent_idx, sentence_tokens in enumerate(sents):
        sent_tokens_lower = [t.lower() for t in sentence_tokens]
        for i in range(len(sent_tokens_lower) - target_len + 1):
            if sent_tokens_lower[i : i + target_len] == target_tokens:
                return {"sent_id": sent_idx, "pos": [i, i + target_len]}

    # --- [2단계] Clean Match (특수기호 무시) ---
    # 원문의 구두점 때문에 토큰이 쪼개진 경우를 대비
    for sent_idx, sentence_tokens in enumerate(sents):
        sent_tokens_clean = [clean_text(t) for t in sentence_tokens]
        
        # 빈 토큰(구두점이었던 것)을 제외하고 인덱스를 추적
        for i in range(len(sent_tokens_clean)):
            # 타겟의 첫 단어와 비슷한 토큰을 찾으면 탐색 시작
            if sent_tokens_clean[i] and sent_tokens_clean[i] == clean_text(target_tokens[0]):
                temp_match = []
                j = i
                target_idx = 0
                
                # 원본 토큰들을 순회하며 target을 완성할 수 있는지 확인
                while j < len(sent_tokens_clean) and target_idx < len(target_tokens):
                    if sent_tokens_clean[j] == clean_text(target_tokens[target_idx]):
                        temp_match.append(j)
                        target_idx += 1
                    elif sent_tokens_clean[j] != "": # 빈 토큰(구두점)이 아닌데 다르면 중단
                        break
                    j += 1
                
                if target_idx == len(target_tokens):
                    return {"sent_id": sent_idx, "pos": [i, temp_match[-1] + 1]}

    # --- [3단계] Fuzzy Match (유사도 검사 - 최후의 보루) ---
    # LLM이 단어 순서를 바꾸거나 형태소를 약간 변형했을 때 사용
    best_match = {"sent_id": -1, "pos": [-1, -1], "score": 0.0}
    
    for sent_idx, sentence_tokens in enumerate(sents):
        # LLM이 요약한 토큰 길이와 비슷할 것으로 예상되는 범위 탐색 (길이 -1 ~ +2)
        for window_size in [target_len - 1, target_len, target_len + 1, target_len + 2]:
            if window_size <= 0: continue
            
            for i in range(len(sentence_tokens) - window_size + 1):
                window_tokens = sentence_tokens[i : i + window_size]
                window_str = " ".join([clean_text(t) for t in window_tokens]).strip()
                
                # difflib을 이용해 문자열 유사도 비율 계산 (0.0 ~ 1.0)
                score = difflib.SequenceMatcher(None, target_clean, window_str).ratio()
                
                if score > best_match["score"]:
                    best_match["score"] = score
                    best_match["sent_id"] = sent_idx
                    best_match["pos"] = [i, i + window_size]

    # 유사도가 80% (0.8) 이상일 때만 유효한 매핑으로 인정
    if best_match["score"] >= 0.80:
        return {"sent_id": best_match["sent_id"], "pos": best_match["pos"]}

    # 3단계까지 모두 실패하면 포기
    return {"sent_id": -1, "pos": [-1, -1]}

# def find_exact_pos_and_sent_id(sents: list, target_text: str) -> dict:
#     """원문 sents 배열에서 특정 텍스트의 sent_id와 pos(토큰 좌표)를 정밀 추적합니다."""
#     target_tokens = [t.lower() for t in target_text.split()]
#     target_len = len(target_tokens)
    
#     for sent_idx, sentence_tokens in enumerate(sents):
#         sent_tokens_lower = [t.lower() for t in sentence_tokens]
#         for i in range(len(sent_tokens_lower) - target_len + 1):
#             if sent_tokens_lower[i : i + target_len] == target_tokens:
#                 return {"sent_id": sent_idx, "pos": [i, i + target_len]}
                
#     return {"sent_id": -1, "pos": [-1, -1]} # 매칭 실패 시

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
        location = find_robust_pos_and_sent_id(sents, entity_name)
        
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
    global current_key_idx
    relation_menu_str = "\n".join([f"       - {pid}: {desc}" for pid, desc in RELATION_MAP.items()])

    prompt = f"""
    You are an Expert Data reconstructor for Knowledge Graphs. 
    Analyze the document and extract core relationships.
    
    [CRITICAL INSTRUCTIONS]

    1. TEMPORAL ENTITY INTEGRITY (STRICT):
    - Do NOT merge entities that represent different historical states/names if the text distinguishes their actions.
    - Every relationship MUST be linked to the EXACT entity name performing the action in its specific sentence.
    - Use inheriting relationships to connect these historical nodes later, but do not consolidate their individual events.

    2. LOCAL SUBJECT PRIORITY:
    - Identify the literal subject (h) and object (t) of the verb in each sentence first. 
    - Do NOT force the {doc_title} as the head if it is not the actual subject of the specific action described.

    3. RELATION SPECIFICITY HIERARCHY:
    - 'r' MUST be a specific action including services or relation, not a vague one. Head and Tail MUST be exact entities from the text that can be explained.
    - 'r' MUST be a Wikidata P-ID which is listed in the menu below. If the relation is not explicitly stated as a P-ID in the text, you MUST infer the MOST appropriate P-ID that is listed.

    4. HYPER-RELATIONAL ANCHORING:
    - Every edge MUST attempt to capture [WHEN] (Time) and [WHY] (Reason/Cause) from the context.
    - If a regulatory action or merger is mentioned, the qualifier is NOT optional.

    5. ENTITY TYPING & NORMALIZATION:
    - Unified naming applies ONLY to identical entities (e.g., "WHO" and "World Health Organization"). 
    - Do NOT unify entities across a timeline where a rebranding or legal change occurred.
    - Do not extract entities that is describing property of another entity. Instance of, part of and other similar descriptive relations should be captured as qualifiers or not included at all.
    - ENTITY TYPING: For every extracted 'h' and 't', you MUST assign one of the following exact types in 'h_type' and 't_type':
        - ORG (Organization / Company / Institution / Department / Government)
        - LOC (Location / Facility)
        - PER (Person)
        - TIME (Date / Year / Duration)
        - NUM (Number / Amount)

    P-ID Relation Menu:
    {relation_menu_str}

    [Document]
    {doc_text}
    """
    
    max_retries = len(API_KEYS)
    attempts = 0

    while attempts < max_retries:
        try:
            # 현재 활성화된 키로 클라이언트 로드
            client = get_current_client()
            
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
            error_msg = str(e).lower()
            
            # 할당량 초과(429) 또는 권한 오류(403) 등 키 변경으로 해결 가능한 에러 감지
            if "429" in error_msg or "quota" in error_msg or "403" in error_msg or "exhausted" in error_msg:
                print(f"\n  ⚠️ [키 {current_key_idx + 1} 한도 초과] 다음 API 키로 전환합니다... ({e})")
                
                # 인덱스를 1 증가시키고, 배열 길이를 넘어가면 다시 0으로 순환
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                attempts += 1
                
                # 전환 후 API 호출 제한을 피하기 위해 약간의 딜레이 추가
                time.sleep(2)
                continue
            elif "503" in error_msg or "unavailable" in error_msg or "overloaded" in error_msg:
                # 시도 횟수가 늘어날수록 5초, 9초, 17초.. 기하급수적으로 대기 시간 증가
                backoff_time = attempts + 5 
                print(f"\n  🚧 [503 서버 혼잡] 잠시 숨을 고릅니다... {backoff_time}초 대기 후 재시도. ({e})")
                attempts += 1
                time.sleep(backoff_time)
                continue
            else:
                # 400 에러(잘못된 프롬프트 포맷) 등은 키를 바꿔도 똑같이 실패하므로 루프 종료
                print(f"\n  ⚠️ [처리 불가능한 API 에러] 키 변경 없이 스킵합니다: {e}")
                return None

    print("\n  ❌ 모든 API 키의 할당량이 소진되었거나 사용 불가능합니다.")
    return None

# --- 5. 메인 처리 로직 ---
def process_split(split_name: str, dataset_path: str, output_base_dir: str):
    dataset = load_from_disk(dataset_path)[split_name]
    os.makedirs(os.path.join(output_base_dir, split_name), exist_ok=True)
    output_file = os.path.join(output_base_dir, split_name, f"{split_name}_enriched_v4.jsonl")
    
    start_idx = sum(1 for _ in open(output_file, 'r', encoding='utf-8')) if os.path.exists(output_file) else 0
    print(f"🚀 [{split_name}] {start_idx}번 문서부터 시작합니다.")

    for i in range(start_idx, start_idx+582): # 테스트 시 range(start_idx, start_idx + 1) 로 변경하세요
        sample = dataset[i]
        doc_text = flatten_document(sample['sents'])
        
        # 💡 수정된 부분: 성공할 때까지 무한 반복하는 while 루프 추가
        while True:
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
                time.sleep(4.5) # 정상 처리 후 다음 문서로 가기 전 대기
                break # 성공했으므로 while 루프를 탈출하고 다음 문서(i+1)로 이동
                
            else:
                # 실패했을 경우 스킵하지 않고 대기 후 루프의 처음으로 돌아가 재시도
                print("❌ (API 실패 - 10초 후 재시도합니다...)")
                time.sleep(10) # 잦은 에러 발생을 막기 위해 10초 휴식 후 재시도

# --- 6. 실행 ---
if __name__ == "__main__":
    SOURCE_PATH = "./Dataset/Re-DocRED"
    OUTPUT_PATH = "./Dataset/HypeRelDocRED"
    
    for split in ["train"]:
        process_split(split, SOURCE_PATH, OUTPUT_PATH)