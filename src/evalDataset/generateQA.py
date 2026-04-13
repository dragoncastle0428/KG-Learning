import os
import json
import time
from dotenv import load_dotenv

# 사용하는 LLM 모델에 맞춰 import 하세요. (여기서는 OpenAI 예시)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. 환경변수 및 LLM 초기화
load_dotenv()
print("🤖 LLM을 초기화합니다...")
# 문서가 500개 이상이므로, 빠르고 저렴한 모델(gpt-4o-mini 등)을 추천합니다.
llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini") 

# 2. 데이터셋 경로 및 설정
DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v2.jsonl"
OUTPUT_PATH = "./evaluation_qa_dataset.json"
DOC_LIMIT = 582 # 추출할 문서 개수 (원하는 만큼 조절)

# 3. LLM에게 지시할 프롬프트 (GraphRAG 평가에 특화된 멀티홉 질문 요구)
prompt = PromptTemplate.from_template("""
당신은 지식 그래프(Knowledge Graph) 시스템의 성능을 평가하는 전문가입니다.
아래 주어진 문서를 읽고, GraphRAG 시스템이 여러 관계(Relationship)를 추론해야만 답할 수 있는 복잡한 질문 1개와 그 정답을 생성하세요.

[조건]
1. 단순한 단어 찾기가 아니라, A와 B의 관계, B와 C의 관계를 알아야 풀 수 있는 형태(Multi-hop reasoning)가 좋습니다.
2. 질문은 한국어로 작성하세요.
3. 정답은 문서 내에 명시된 짧고 명확한 단어(엔티티) 위주로 작성하세요.
4. 반드시 아래의 JSON 포맷으로만 출력하세요. 마크다운(` ```json ` 등)을 포함하지 마세요.

[포맷]
{{
    "question": "이곳에 생성된 질문 작성",
    "answer": "이곳에 정답 작성"
}}

[문서]
{text}
""")

chain = prompt | llm

print(f"📖 {DATASET_PATH} 에서 텍스트를 읽어옵니다...")
qa_dataset = []

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for doc_id, line in enumerate(f):
        if doc_id >= DOC_LIMIT:
            break
            
        data = json.loads(line)
        
        # Re-DocRED 데이터셋의 'sents' (문장 리스트)를 하나의 텍스트로 합치기
        # sents 구조: [['단어', '단어', ...], ['단어', '단어', ...]]
        sentences = [" ".join(sent) for sent in data.get('sents', [])]
        document_text = " ".join(sentences)
        
        # 텍스트가 너무 짧으면 건너뛰기
        if len(document_text) < 50:
            continue
            
        print(f"진행 중... [{doc_id+1}/{DOC_LIMIT}] 문서의 QA 생성 중")
        
        try:
            # LLM에게 질문 생성 요청
            response = chain.invoke({"text": document_text})
            output_text = response.content.strip()
            
            # 마크다운 찌꺼기 제거 (LLM이 가끔 json 블록을 넣을 때 대비)
            if output_text.startswith("```json"):
                output_text = output_text[7:-3].strip()
            elif output_text.startswith("```"):
                output_text = output_text[3:-3].strip()
                
            qa_pair = json.loads(output_text)
            
            # doc_id와 원본 텍스트도 평가를 위해 함께 저장해둡니다.
            qa_dataset.append({
                "doc_id": doc_id,
                "context": document_text,
                "question": qa_pair["question"],
                "answer": qa_pair["answer"]
            })
            
            # API Rate Limit(호출 제한) 방지를 위해 약간의 대기 시간 추가
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"문서 {doc_id} 처리 중 에러 발생: {e}")

# 4. 생성된 QA 데이터셋을 JSON 파일로 저장
print(f"\n🎉 총 {len(qa_dataset)}개의 평가용 질문이 생성되었습니다!")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
    json.dump(qa_dataset, out_f, ensure_ascii=False, indent=4)

print(f"💾 데이터가 '{OUTPUT_PATH}' 에 안전하게 저장되었습니다.")