import os
import json
import time
import random # 무작위 추출을 위한 모듈 추가
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

print("💎 Gemini 모델을 초기화합니다...")

SELECTED_API_KEY = os.getenv("GEMINI_API_KEY30") 

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=SELECTED_API_KEY
)

DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v3.jsonl"
OUTPUT_PATH = "./results/graphrag_evaluation_qa.json"

# 무료 티어 제한을 고려하여 뽑아낼 '랜덤 문서'의 총 개수 설정 (예: 10개)
SAMPLE_SIZE = 15

prompt = PromptTemplate.from_template("""
당신은 지식 그래프(Knowledge Graph) 시스템의 추론 능력을 평가하는 출제 위원입니다.
주어진 문서를 읽고, 일반적인 검색으로는 풀기 어렵고 '여러 엔티티 간의 관계를 타고 넘어가야(Multi-hop)' 풀 수 있는 복잡한 질문과 정답을 2~3개 생성하세요.

[출제 원칙 - 매우 중요]
1. Multi-hop: A와 B의 관계, B와 C의 관계를 모두 알아야 대답할 수 있도록 꼬리를 무는 질문을 만드세요. 
2. Qualifier 활용: 문서에 날짜, 시간, 직책 등 구체적인 조건이 있다면 이를 질문에 적극 포함하세요.
3. 정답은 문장형이 아니라, 지식 그래프의 '노드(엔티티)' 형태로 짧게 떨어지는 명사여야 합니다.
4. 반드시 영어로 작성하고, 마크다운(```json) 없이 순수한 JSON 배열(List) 포맷만 출력하세요.

[포맷 - 반드시 이 배열 형태를 유지하세요]
[
    {{
        "question": "이곳에 첫 번째 복잡한 질문 작성",
        "answer": "첫 번째 정답 엔티티"
    }},
    {{
        "question": "이곳에 두 번째 복잡한 질문 작성",
        "answer": "두 번째 정답 엔티티"
    }},
    {{
        "question": "이곳에 세 번째 복잡한 질문 작성",
        "answer": "세 번째 정답 엔티티"
    }}
]

[문서]
{text}
""")

chain = prompt | llm

print(f"📖 {DATASET_PATH} 에서 텍스트를 읽어옵니다...")
qa_dataset = []

# 1. 파일의 모든 줄을 읽어와서 리스트에 담습니다. (원본 doc_id 유지를 위해 enumerate 사용)
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    all_docs = list(enumerate(f))

# 2. 전체 문서 중에서 SAMPLE_SIZE 만큼 무작위로 뽑아냅니다.
# (전체 문서가 SAMPLE_SIZE보다 적을 경우를 대비해 min 처리)
sampled_docs = random.sample(all_docs, min(SAMPLE_SIZE, len(all_docs)))

print(f"🎲 전체 {len(all_docs)}개 문서 중 {len(sampled_docs)}개를 무작위로 선정하여 출제를 시작합니다.\n")

# 3. 무작위로 뽑힌 문서들만 순회하며 질문을 생성합니다.
for i, (original_doc_id, line) in enumerate(sampled_docs):
    data = json.loads(line)
    sentences = [" ".join(sent) for sent in data.get('sents', [])]
    document_text = " ".join(sentences)
    
    if len(document_text) < 100: 
        print(f"진행 중... [{i+1}/{SAMPLE_SIZE}] 문서가 너무 짧아 건너뜁니다. (Original ID: {original_doc_id})")
        continue
        
    print(f"진행 중... [{i+1}/{SAMPLE_SIZE}] 복잡한 QA 2~3개 생성 중 (Original ID: {original_doc_id})")
    
    try:
        response = chain.invoke({"text": document_text})
        output_text = response.content.strip()
        
        # 마크다운 찌꺼기 제거
        if output_text.startswith("```json"):
            output_text = output_text[7:-3].strip()
        elif output_text.startswith("```"):
            output_text = output_text[3:-3].strip()
            
        # 리스트 형태로 파싱
        qa_list = json.loads(output_text)
        
        # 뽑혀 나온 질문 2~3개를 순회하면서 개별 데이터로 저장
        for qa_pair in qa_list:
            qa_dataset.append({
                "doc_id": original_doc_id, # 나중에 추적하기 쉽도록 원본 ID 저장
                "question": qa_pair.get("question", ""),
                "answer": qa_pair.get("answer", ""),
                "context": document_text
            })
        
        # 무료 티어 Rate Limit (15 RPM) 안전장치: 1분에 최대 12개 정도만 처리하도록 5초 딜레이
        time.sleep(5) 
        
    except Exception as e:
        print(f"문서 {original_doc_id} 처리 중 에러 발생 (스킵): {e}\n내용: {output_text[:50]}...")

# 4. 저장
print(f"\n🎉 무작위 {len(sampled_docs)}개의 문서에서 총 {len(qa_dataset)}개의 GraphRAG 평가용 질문이 생성되었습니다!")

# 결과 폴더가 없으면 에러가 나므로, 폴더 자동 생성 로직 추가
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
    json.dump(qa_dataset, out_f, ensure_ascii=False, indent=4)

print(f"💾 데이터가 '{OUTPUT_PATH}' 에 안전하게 저장되었습니다.")