import os
import json
import random
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain & Neo4j
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate  # 👈 추가된 임포트

# Google GenAI (질문 생성 및 평가용)
from google.genai import Client
from google.genai import types

# 1. 환경 변수 및 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
V4_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl"
SAMPLE_DOCS_COUNT = 3  # 질문을 추출할 문서 개수

print("🔌 데이터베이스 연결 중...")
graph_orig = Neo4jGraph(database="original") 
graph_hyper = Neo4jGraph(database="hyper")
print("✅ DB 연결 완료!")

# 2. RAG 체인 설정 (Cypher 생성용)
llm_rag = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

# Original DB 체인은 기본 프롬프트 사용
chain_orig = GraphCypherQAChain.from_llm(
    llm_rag, 
    graph=graph_orig, 
    verbose=False, 
    allow_dangerous_requests=True
)

# --- 🌟 추가된 부분: Hyper 전용 넓은 검색(Broad Subgraph) 프롬프트 ---
HYPER_CYPHER_TEMPLATE = """
You are an expert Neo4j Cypher translator.
Your goal is to extract a rich subgraph that contains the answer to the user's question.

Schema:
{schema}

[CRITICAL INSTRUCTIONS for Hyper Knowledge Graph]
1. DO NOT try to guess or filter by specific relationship types.
2. Instead, identify the core entities in the question, and use a broad `CONTAINS` match with `toLower()` to find them.
3. Once the core entities are found, match ALL their connected relationships (up to 1 or 2 hops).
4. You MUST use `properties(r)` to return ALL relationship properties. This is extremely important because properties contain the Qualifiers (WHAT, WHY, WHERE, WHEN, HOW) needed to answer complex questions.

[Example Query Structure]
Question: 데스 망간이 노르웨이에 돈을 제안한 이유는 무엇인가요?
Cypher:
MATCH (n:Entity)-[r]-(m:Entity)
WHERE toLower(n.name) CONTAINS 'mangan' OR toLower(n.name) CONTAINS 'norway'
RETURN n.name, type(r), properties(r), m.name
LIMIT 30

Return ONLY the valid Cypher query without any markdown formatting or explanations.

Question: {question}
"""

hyper_prompt = PromptTemplate(
    template=HYPER_CYPHER_TEMPLATE,
    input_variables=["schema", "question"]
)

# Hyper DB 체인에 커스텀 프롬프트 주입 및 verbose 활성화
chain_hyper = GraphCypherQAChain.from_llm(
    llm_rag, 
    graph=graph_hyper, 
    verbose=True,  # 👈 어떤 쿼리를 생성하는지 콘솔에서 확인 가능
    allow_dangerous_requests=True,
    cypher_prompt=hyper_prompt
)
# -----------------------------------------------------------

# 3. Pydantic 스키마 정의 (질문 생성용 & 평가용)
class GeneratedQuestion(BaseModel):
    question_type: str = Field(description="'Simple(단일 관계)', 'Multi-hop(연쇄 관계)', 'Qualifier(조건/이유/장소 등 부가정보 포함)' 중 하나")
    question: str = Field(description="RAG 시스템에 던질 자연어 질문 (한국어)")
    expected_focus: str = Field(description="이 질문을 통해 확인하고자 하는 핵심 정보 (예: BGC의 운송 목적지 및 이유)")

class QuestionList(BaseModel):
    questions: list[GeneratedQuestion]

class RAGEvaluation(BaseModel):
    original_score: float = Field(ge=0.0, le=10.0)
    hyper_score: float = Field(ge=0.0, le=10.0)
    winner: str = Field(description="'Original', 'Hyper', 또는 'Tie'")
    reason: str = Field(description="점수 부여 및 승자 선정의 핵심 이유 (한국어)")

# 4. 헬퍼 함수
def get_doc_and_graph(doc_data):
    doc_text = " ".join([" ".join(sent) for sent in doc_data.get('sents', [])])
    graph_info = json.dumps(doc_data.get('labels', []), ensure_ascii=False) 
    return doc_text, graph_info

# 5. 질문 자동 생성 엔진
def generate_questions_for_doc(doc_text, graph_info):
    client = Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    당신은 RAG 시스템의 성능을 평가하기 위한 고급 질문 출제자입니다.
    아래 주어진 원문(Document)과 추출된 지식 그래프 정보(Graph Info, 특히 Qualifiers)를 바탕으로,
    AI가 그래프를 잘 탐색하는지 테스트할 수 있는 질문 3개를 생성하세요.

    [요구조건]
    1. Simple: 두 엔티티 간의 1차원적인 관계를 묻는 질문 1개.
    2. Multi-hop: A -> B -> C 로 이어지는 연쇄적인 관계를 추론해야 하는 질문 1개.
    3. Qualifier: '왜', '어디서', '어떻게', '무엇을' 등 엣지의 부가정보(Qualifier)를 알아야만 답할 수 있는 질문 1개.

    [데이터]
    원문: {doc_text}
    그래프 정보(참고용): {graph_info}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=QuestionList,
                temperature=0.7,
            ),
        )
        return response.parsed.questions
    except Exception as e:
        print(f"질문 생성 실패: {e}")
        return []

# 6. RAG 평가 엔진
def evaluate_rag_responses(question, expected_focus, ans_orig, ans_hyper):
    client = Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    당신은 AI QA 시스템의 성능을 평가하는 심사위원입니다.

    [질문]: {question}
    [채점 핵심 기준]: {expected_focus}

    [Original DB RAG 답변]
    {ans_orig}

    [Hyper DB RAG 답변]
    {ans_hyper}

    두 답변을 비교하여 '채점 핵심 기준'을 잘 충족했는지, 정보가 얼마나 풍부하고 정확한지 평가하세요.
    만약 '모른다'고 대답했거나 엉뚱한 쿼리를 생성한 듯한 답변은 감점하세요.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json',
                response_schema=RAGEvaluation,
                temperature=0.1,
            ),
        )
        return response.parsed
    except Exception as e:
        return None

# --- 실행 파트 ---
if __name__ == "__main__":
    # 데이터 로드
    with open(V4_DATASET_PATH, 'r', encoding='utf-8') as f:
        all_docs = [json.loads(line) for line in f]
    
    # 샘플 문서 무작위 추출
    sample_docs = random.sample(all_docs, min(SAMPLE_DOCS_COUNT, len(all_docs)))
    
    results = []
    print("\n📝 1단계: 데이터셋 기반 맞춤형 평가 질문 생성 중...\n")
    
    test_questions = []
    for doc in sample_docs:
        doc_text, graph_info = get_doc_and_graph(doc)
        generated_qs = generate_questions_for_doc(doc_text, graph_info)
        test_questions.extend(generated_qs)
        
    print(f"총 {len(test_questions)}개의 고품질 질문이 생성되었습니다!")
    for q in test_questions:
        print(f" - [{q.question_type}] {q.question}")
        
    print("\n🚀 2단계: GraphRAG 성능 비교 테스트 시작...\n")
    
    for i, q_obj in enumerate(test_questions):
        query = q_obj.question
        print(f"\n[{i+1}/{len(test_questions)}] {query} (유형: {q_obj.question_type})")
        
        # Original 테스트
        try:
            ans_orig = chain_orig.invoke({"query": query})["result"]
        except Exception:
            ans_orig = "오류 발생"
            
        # Hyper 테스트
        try:
            ans_hyper = chain_hyper.invoke({"query": query})["result"]
        except Exception:
            ans_hyper = "오류 발생"
            
        print(f" └ Original: {ans_orig}")
        print(f" └ Hyper   : {ans_hyper}")
        
        # LLM 평가
        eval_result = evaluate_rag_responses(query, q_obj.expected_focus, ans_orig, ans_hyper)
        if eval_result:
            print(f" 🏆 Winner: {eval_result.winner} (Orig {eval_result.original_score} vs Hyper {eval_result.hyper_score})")
            print(f" 💡 Reason: {eval_result.reason}")
            
            results.append({
                "Question_Type": q_obj.question_type,
                "Question": query,
                "Expected_Focus": q_obj.expected_focus,
                "Original_Answer": ans_orig,
                "Hyper_Answer": ans_hyper,
                "Original_Score": eval_result.original_score,
                "Hyper_Score": eval_result.hyper_score,
                "Winner": eval_result.winner,
                "Reason": eval_result.reason
            })

    # 결과 저장
    if results:
        df = pd.DataFrame(results)
        filename = f"graph_rag_dynamic_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\n✅ 동적 비교 평가 완료! 결과가 '{filename}'에 저장되었습니다.")