import os
import json
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datasets import load_from_disk
from google.genai import Client
from google.genai import types
from pydantic import BaseModel, Field

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic' # Mac은 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 설정 및 다중 API 키 초기화 ---
load_dotenv()
API_KEYS = [os.getenv(f"GEMINI_API_KEY{i}") for i in range(25, 31) if os.getenv(f"GEMINI_API_KEY{i}")]
if not API_KEYS:
    raise ValueError("설정된 API 키가 없습니다. .env 파일을 확인해주세요.")

current_key_idx = 0

def get_current_client():
    return Client(api_key=API_KEYS[current_key_idx])

ORIGINAL_DATASET_PATH = "./Dataset/Re-DocRED"
NEW_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v2.jsonl"
SAMPLE_SIZE = 50 # 50개 평가

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
    "P1056": "service provided", "P1444": "destination point", "P3608": "regulated by"
}

# --- 2. 출력 스키마 정의 (0.0 ~ 100.0 소수점 허용) ---
class ScoreDetail(BaseModel):
    score: float = Field(ge=0.0, le=100.0, description="Score from 0.0 to 100.0 (allows 1 decimal place)")
    reason: str = Field(description="Short rationale for the score in Korean")

class DatasetEval(BaseModel):
    hallucination: ScoreDetail = Field(description="사실 부합성 (0.0: 치명적 환각 ~ 100.0: 원문 완벽 기반)")
    comprehensiveness: ScoreDetail = Field(description="정보 포괄성/풍부함 (0.0: 빈약함 ~ 100.0: 완벽한 디테일)")
    schema_mapping: ScoreDetail = Field(description="스키마 매핑 정확도 (0.0: 엉뚱함 ~ 100.0: 완벽한 매핑)")
    salience: ScoreDetail = Field(description="핵심성 (0.0: TMI 위주 ~ 100.0: 핵심 주제 위주)")
    entity_clarity: ScoreDetail = Field(description="엔티티 명확성/통합 (0.0: 파편화됨 ~ 100.0: 고유명사로 완벽히 통합됨)")

class JudgeResult(BaseModel):
    original_score: DatasetEval
    new_score: DatasetEval
    overall_verdict: str = Field(description="어느 데이터셋이 더 나은지에 대한 총평 (Korean)")

# --- 3. 데이터 헬퍼 함수 ---
def load_new_dataset(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def format_graph_for_prompt(vertex_set, labels):
    if not vertex_set or not labels:
        return "No relations extracted."
    
    output = []
    node_dict = {i: v_group[0]['name'] for i, v_group in enumerate(vertex_set)}
    
    for lbl in labels:
        h_name = node_dict.get(lbl['h'], "Unknown")
        t_name = node_dict.get(lbl['t'], "Unknown")
        r_name = RELATION_MAP.get(lbl['r'], lbl['r'])
        
        quals = ""
        if 'qualifiers' in lbl and lbl['qualifiers']:
            quals_list = [f"{q.get('qualifier', '')}:{q.get('qualifier_value', '')}" for q in lbl['qualifiers']]
            quals = f" (Details: {', '.join(quals_list)})"
            
        output.append(f"- [{h_name}] --({r_name})--> [{t_name}]{quals}")
        
    return "\n".join(output)

# --- 4. 시각화 함수 ---
def plot_evaluation_results(orig_avgs, new_avgs, metrics):
    """5가지 평가 항목에 대한 0-100점 결과를 그룹 막대 그래프로 시각화합니다."""
    # 보기 좋은 이름으로 포맷팅 (예: entity_clarity -> Entity Clarity)
    display_labels = [m.replace('_', ' ').title() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    orig_scores = [orig_avgs[m] for m in metrics]
    new_scores = [new_avgs[m] for m in metrics]

    bars1 = ax.bar(x - width/2, orig_scores, width, label='Re-DocRED (Original)', color='#aab7b8')
    bars2 = ax.bar(x + width/2, new_scores, width, label='HypeRelDocRED (New)', color='#2980b9')

    ax.set_ylabel('Average Score (0 - 100)', fontsize=12)
    ax.set_title(f'LLM-as-a-Judge Quality Evaluation (n={SAMPLE_SIZE})', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_ylim(0, 110) # 텍스트가 잘리지 않도록 y축을 110까지 여유있게 설정
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # 막대 위에 소수점 1자리까지 점수 텍스트 추가
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 수직으로 5포인트 띄움
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --- 5. 다중 키 대응 LLM 평가 호출 함수 ---
def evaluate_document(doc_text: str, orig_graph_str: str, new_graph_str: str) -> JudgeResult:
    global current_key_idx
    prompt = f"""
    당신은 지식 그래프 데이터셋의 품질을 평가하는 엄격하고 정밀한 심사위원입니다.
    아래 [원본 문서]와 두 가지 버전의 지식 그래프를 비교 평가하세요.
    
    [평가 기준] (0.0점: 최악 ~ 100.0점: 완벽, 소수점 첫째 자리까지 세밀하게 평가)
    1. 환각의 정도 (Hallucination): 원문에 없는 지식 창작 및 왜곡 여부 (100점: 완벽히 원문 기반)
    2. 정보 포괄성 (Comprehensiveness): 원문 속 디테일(시간, 장소 등)의 풍부함 (100점: 풍부하고 완벽함)
    3. 스키마 매핑 정확도 (Schema Mapping): 적절한 P-ID 매핑 여부 (100점: 가장 정확한 관계 매핑)
    4. 핵심성 (Salience): 핵심 주제 중심의 추출 여부 (100점: 문서의 핵심을 완벽히 관통함)
    5. 엔티티 명확성 (Entity Clarity): 대명사 배제 및 고유명사 통합 여부 (100점: 완벽히 정제/통합됨)
    
    [Original Document]
    {doc_text}
    
    [Original Dataset (Re-DocRED)]
    {orig_graph_str}
    
    [New Dataset (HypeRelDocRED)]
    {new_graph_str}
    
    출력은 반드시 제공된 JSON 스키마를 엄격히 따르며, 이유는 한국어로 작성하세요.
    """
    
    max_attempts = len(API_KEYS) * 3
    attempts = 0

    while attempts < max_attempts:
        try:
            client = get_current_client()
            response = client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=JudgeResult,
                    temperature=0, # 평가의 일관성을 위해 0으로 유지
                ),
            )
            return response.parsed
            
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                print(f"  ⚠️ [키 {current_key_idx + 1} 한도 초과] 다음 키로 전환합니다...")
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                time.sleep(2)
            elif "503" in error_msg or "unavailable" in error_msg:
                backoff = attempts + 5
                print(f"  🚧 [서버 혼잡] {backoff}초 대기 후 재시도...")
                time.sleep(backoff)
            else:
                print(f"  ❌ 치명적 에러 발생: {e}")
                return None
        attempts += 1

    print("  ❌ 모든 API 키의 할당량이 소진되었거나 응답이 불가능합니다.")
    return None

# --- 6. 메인 실행 루프 ---
if __name__ == "__main__":
    print(f"🚀 총 {len(API_KEYS)}개의 API 키가 로드되었습니다.")
    print(f"데이터셋 로드 및 {SAMPLE_SIZE}개 샘플링 시작...\n")
    
    original_data = [doc for doc in load_from_disk(ORIGINAL_DATASET_PATH)["train"]]
    new_data = load_new_dataset(NEW_DATASET_PATH)
    
    min_len = min(len(original_data), len(new_data))
    sample_indices = random.sample(range(min_len), SAMPLE_SIZE)
    
    results = []
    success_count = 0
    
    for idx, i in enumerate(sample_indices):
        print(f"[{success_count + 1}/{SAMPLE_SIZE}] 문서 ID {i} 평가 중...", end=" ", flush=True)
        
        orig_doc = original_data[i]
        new_doc = new_data[i]
        
        doc_text = " ".join([" ".join(sent) for sent in orig_doc['sents']])
        orig_graph_str = format_graph_for_prompt(orig_doc.get('vertexSet', []), orig_doc.get('labels', []))
        new_graph_str = format_graph_for_prompt(new_doc.get('vertexSet', []), new_doc.get('labels', []))
        
        eval_result = evaluate_document(doc_text, orig_graph_str, new_graph_str)
        
        if eval_result:
            results.append(eval_result)
            success_count += 1
            print("✅ 완료")
            time.sleep(3.5)
        else:
            print("건너뜀")

    # --- 7. 결과 집계 및 시각화 ---
    if results:
        print("\n" + "="*65)
        print(" 🏆 최종 품질 평가 결과 (평균 점수, 100점 만점)")
        print("="*65)
        
        metrics = ['hallucination', 'comprehensiveness', 'schema_mapping', 'salience', 'entity_clarity']
        orig_avgs = {m: sum(getattr(r.original_score, m).score for r in results) / len(results) for m in metrics}
        new_avgs = {m: sum(getattr(r.new_score, m).score for r in results) / len(results) for m in metrics}
        
        print(f"{'평가 항목':<20} | {'Original (Re-DocRED)':<20} | {'New (HypeRelDocRED)':<20}")
        print("-" * 65)
        for m in metrics:
            print(f"{m:<20} | {orig_avgs[m]:<20.1f}점 | {new_avgs[m]:<20.1f}점")
            
        with open("evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in results], f, ensure_ascii=False, indent=2)
        print("\n✅ 전체 평가 상세 결과가 'evaluation_results.json'에 저장되었습니다.")
        
        # 📊 시각화 창 띄우기
        print("\n📊 평가 결과 차트를 생성합니다...")
        plot_evaluation_results(orig_avgs, new_avgs, metrics)
    else:
        print("\n평가를 완료한 문서가 없습니다.")