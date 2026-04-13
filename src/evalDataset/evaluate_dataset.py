import os
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # 엑셀 저장을 위해 추가
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_from_disk
from google.genai import Client
from google.genai import types
from pydantic import BaseModel, Field

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 설정 및 경로 ---
load_dotenv()
API_KEYS = [os.getenv(f"GEMINI_API_KEY{i}") for i in range(1, 31) if os.getenv(f"GEMINI_API_KEY{i}")]
if not API_KEYS:
    raise ValueError("설정된 API 키가 없습니다.")

current_key_idx = 0

ORIGINAL_DATASET_PATH = "./Dataset/Re-DocRED"
V2_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v2.jsonl" # V2 경로
V3_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v3.jsonl" # V3 경로
SAMPLE_SIZE = 50 

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
    "P6": "head of government", "P27": "country of citizenship", "P30": "continent",
    "P31": "instance of", "P35": "head of state", "P37": "official language",
    "P39": "position held", "P40": "child", "P54": "member of sports team",
    "P58": "screenwriter", "P86": "composer", "P102": "member of political party",
    "P118": "league", "P123": "publisher", "P136": "genre", "P137": "operator",
    "P140": "religion", "P150": "contains administrative territorial entity",
    "P161": "cast member", "P162": "producer", "P170": "creator",
    "P171": "parent taxon", "P172": "ethnic group", "P176": "manufacturer",
    "P178": "developer", "P179": "series", "P194": "legislative body",
    "P205": "basin country", "P206": "located in or next to body of water",
    "P241": "military branch", "P264": "record label", "P272": "production company",
    "P276": "location", "P279": "subclass of", "P361": "part of",
    "P364": "original language", "P400": "platform", "P403": "mouth of the watercourse",
    "P449": "original network", "P463": "member of", "P488": "chair",
    "P495": "country of origin", "P527": "has part", "P551": "residence",
    "P576": "dissolved or abolished date", "P580": "start time", "P582": "end time",
    "P607": "conflict", "P674": "characters", "P676": "lyrics by",
    "P706": "located on terrain feature", "P710": "participant",
    "P737": "influenced by", "P740": "location of formation", "P800": "notable work",
    "P807": "separated from", "P840": "narrative location", "P937": "work location",
    "P1001": "legal jurisdiction", "P1198": "unemployment rate",
    "P1336": "territory claimed by", "P1344": "participant in",
    "P1376": "capital of", "P1412": "languages spoken or written",
    "P1441": "present in work", "P3373": "sibling",
}

# --- 2. 스키마 정의 ---
class ScoreDetail(BaseModel):
    score: float = Field(ge=0.0, le=100.0)
    reason: str 

class DatasetEval(BaseModel):
    hallucination: ScoreDetail
    comprehensiveness: ScoreDetail
    schema_mapping: ScoreDetail
    salience: ScoreDetail
    entity_clarity: ScoreDetail

class JudgeResult(BaseModel):
    original_score: DatasetEval
    v2_score: DatasetEval
    v3_score: DatasetEval
    overall_verdict: str = Field(description="3개 데이터셋의 차이점과 v3의 개선 핵심 요약 (Korean)")

# --- 3. 헬퍼 함수 ---
def get_current_client():
    return Client(api_key=API_KEYS[current_key_idx])

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def format_graph_for_prompt(vertex_set, labels):
    if not vertex_set or not labels: return "No relations extracted."
    output = []
    node_dict = {i: v_group[0]['name'] for i, v_group in enumerate(vertex_set)}
    for lbl in labels:
        h_name, t_name = node_dict.get(lbl['h'], "Unknown"), node_dict.get(lbl['t'], "Unknown")
        r_name = RELATION_MAP.get(lbl['r'], lbl['r'])
        quals = ""
        if 'qualifiers' in lbl and lbl['qualifiers']:
            q_list = [f"{q.get('qualifier','')}:{q.get('qualifier_value','')}" for q in lbl['qualifiers']]
            quals = f" (Details: {', '.join(q_list)})"
        output.append(f"- [{h_name}] --({r_name})--> [{t_name}]{quals}")
    return "\n".join(output)

# --- 4. 시각화 (3개 그룹) ---
def plot_evaluation_results(orig_avgs, v2_avgs, v3_avgs, metrics):
    display_labels = [m.replace('_', ' ').title() for m in metrics]
    x = np.arange(len(metrics))
    width = 0.25 # 막대 너비 조절

    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - width, [orig_avgs[m] for m in metrics], width, label='Original (Re-DocRED)', color='#bdc3c7')
    bars2 = ax.bar(x, [v2_avgs[m] for m in metrics], width, label='V2 (Enriched)', color='#3498db')
    bars3 = ax.bar(x + width, [v3_avgs[m] for m in metrics], width, label='V3 (Latest)', color='#e67e22')

    ax.set_ylabel('Average Score (0 - 100)')
    ax.set_title(f'Comparative Quality Evaluation: Original vs V2 vs V3 (n={SAMPLE_SIZE})', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend()

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(bars1); autolabel(bars2); autolabel(bars3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 5. 평가 엔진 ---
def evaluate_document(doc_text, g_orig, g_v2, g_v3):
    global current_key_idx
    
    prompt = f"""
    당신은 지식 추출(Information Extraction) 데이터의 미세한 품질 차이를 식별하는 정밀 심사위원입니다.
    제공된 [Original Document]를 기준으로 세 가지 지식 그래프의 품질을 독립적으로 평가하십시오.

    ### [평가 원칙: 세밀한 변별력 확보]
    - **연속적 점수 부여**: 5점이나 10점 단위로 끊어지는 점수보다는, 데이터의 완성도를 정교하게 반영한 소수점 단위 점수(예: 87.3, 92.6)를 권장합니다.
    - **가중치 기반 정성 평가**: 가이드라인은 심각도를 판단하는 '기준'일 뿐이며, 오류의 맥락, 문서 내 중요도, 추출된 정보의 난이도를 종합적으로 고려하여 최종 점수를 산출하십시오.

    ### [평가 지표 및 판단 기준]
    1. **Hallucination (환각)**: 원문에 없는 정보나 왜곡된 관계 포함 여부.
    2. **Comprehensiveness (정보 포괄성)**: 주요 관계 및 세부 정보(날짜, 속성 등)의 풍부함.
    3. **Schema Mapping (스키마 정확도)**: P-ID가 관계의 시맨틱을 얼마나 정밀하게 표현하는가.
    4. **Salience (핵심성)**: 문서의 중심 사건이 KG의 중심축을 이루는가.
    5. **Entity Clarity (엔티티 명확성)**: 대명사 해결 및 엔티티 통합의 정교함.

    ---
    ### [평가 대상 데이터]
    [Original Document]
    {doc_text}

    [1. Original (Re-DocRED)]
    {g_orig}

    [2. Version 2 (HypeRelDocRED v2)]
    {g_v2}

    [3. Version 3 (HypeRelDocRED v3)]
    {g_v3}

    ---
    ### [출력 요구사항]
    - 반드시 제공된 JSON 스키마(`JudgeResult`)를 엄격히 따르십시오.
    - `reason`은 한국어로 작성하며, 점수 산정의 결정적 근거를 기술하십시오.
    """

    max_retries = len(API_KEYS) * 2  # 모든 키를 두 번씩 돌 때까지 재시도
    retry_count = 0

    while retry_count < max_retries:
        try:
            client = get_current_client()
            response = client.models.generate_content(
                model='gemini-2.5-flash', # 혹은 gemini-1.5-flash
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=JudgeResult,
                    temperature=0.1, # 약간의 변별력을 위해 0.1 권장
                ),
            )
            return response.parsed

        except Exception as e:
            # 429 에러(Quota Exceeded) 발생 시
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                print(f"\n[!] Rate Limit Reached. Switching to API Key {current_key_idx + 1}/{len(API_KEYS)}...")
                
                # 모든 키를 한 바퀴 다 썼다면 잠시 대기
                if current_key_idx == 0:
                    print("[!] All keys exhausted. Cooling off for 30 seconds...")
                    time.sleep(30)
                
                retry_count += 1
                continue # 다음 키로 즉시 재시도
            
            else:
                print(f"\n[!] Unexpected Error: {e}")
                time.sleep(5)
                retry_count += 1
                
    print("\n[!] Failed to evaluate document after maximum retries.")
    return None

def save_to_files(results, orig_avgs, v2_avgs, v3_avgs, metrics):
    """평가 결과 데이터를 엑셀과 PDF(이미지 포함)로 저장합니다."""
    
    # 1. 엑셀 파일 생성 (상세 내역)
    rows = []
    for i, res in enumerate(results):
        data = {"sample_index": i}
        for m in metrics:
            data[f"orig_{m}"] = getattr(res.original_score, m).score
            data[f"v2_{m}"] = getattr(res.v2_score, m).score
            data[f"v3_{m}"] = getattr(res.v3_score, m).score
        data["overall_verdict"] = res.overall_verdict
        rows.append(data)
    
    df = pd.DataFrame(rows)
    excel_filename = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(excel_filename, index=False)
    print(f"\n[+] 상세 데이터가 {excel_filename}으로 저장되었습니다.")

    # 2. 시각화 결과 이미지 저장 및 출력
    display_labels = [m.replace('_', ' ').title() for m in metrics]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width, [orig_avgs[m] for m in metrics], width, label='Original', color='#bdc3c7')
    ax.bar(x, [v2_avgs[m] for m in metrics], width, label='V2', color='#3498db')
    ax.bar(x + width, [v3_avgs[m] for m in metrics], width, label='V3', color='#e67e22')

    ax.set_ylabel('Average Score')
    ax.set_title('Comparative Quality Evaluation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.legend()
    
    plot_filename = "eval_plot.png"
    plt.savefig(plot_filename)
    plt.show()
    print(f"[+] 시각화 그래프가 {plot_filename}으로 저장되었습니다.")

# --- 6. 실행 ---
if __name__ == "__main__":
    original_data = [doc for doc in load_from_disk(ORIGINAL_DATASET_PATH)["train"]]
    v2_data = load_jsonl(V2_DATASET_PATH)
    v3_data = load_jsonl(V3_DATASET_PATH)
    
    min_len = min(len(original_data), len(v2_data), len(v3_data))
    sample_indices = random.sample(range(min_len), SAMPLE_SIZE)
    
    results = []
    metrics = ['hallucination', 'comprehensiveness', 'schema_mapping', 'salience', 'entity_clarity']

    for i in sample_indices:
        doc_text = " ".join([" ".join(sent) for sent in original_data[i]['sents']])
        g_orig = format_graph_for_prompt(original_data[i].get('vertexSet'), original_data[i].get('labels'))
        g_v2 = format_graph_for_prompt(v2_data[i].get('vertexSet'), v2_data[i].get('labels'))
        g_v3 = format_graph_for_prompt(v3_data[i].get('vertexSet'), v3_data[i].get('labels'))
        
        print(f"Evaluating Sample {i}...", end="\r")
        res = evaluate_document(doc_text, g_orig, g_v2, g_v3)
        if res: results.append(res)
        time.sleep(2)

    # 결과 집계
    if results:
        orig_avgs = {m: sum(getattr(r.original_score, m).score for r in results)/len(results) for m in metrics}
        v2_avgs = {m: sum(getattr(r.v2_score, m).score for r in results)/len(results) for m in metrics}
        v3_avgs = {m: sum(getattr(r.v3_score, m).score for r in results)/len(results) for m in metrics}
        
        # 테이블 출력 및 시각화 호출
        print("\n" + "="*80)
        print(f"{'Metric':<20} | {'Original':<15} | {'V2':<15} | {'V3':<15}")
        print("-" * 80)
        for m in metrics:
            print(f"{m:<20} | {orig_avgs[m]:<15.1f} | {v2_avgs[m]:<15.1f} | {v3_avgs[m]:<15.1f}")
        
        plot_evaluation_results(orig_avgs, v2_avgs, v3_avgs, metrics)
        save_to_files(results, orig_avgs, v2_avgs, v3_avgs, metrics)