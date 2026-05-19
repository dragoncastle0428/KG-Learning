import os
import json
import random
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_community.graphs import Neo4jGraph
from google.genai import Client
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
V4_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl"
SAMPLE_DOCS_COUNT = 3
MAX_SUBGRAPH_RESULTS = 30

print("🔌 데이터베이스 연결 중...")
graph_orig  = Neo4jGraph(database="original")
graph_hyper = Neo4jGraph(database="hyper")
print("✅ DB 연결 완료!")

client = Client(api_key=GEMINI_API_KEY)

# ─────────────────────────────────────────────
# 1. Pydantic 스키마
# ─────────────────────────────────────────────
class ExtractedEntities(BaseModel):
    entities: List[str] = Field(description="Key entity names extracted from the question")

class GeneratedQuestion(BaseModel):
    question_type: str = Field(description="'Simple', 'Multi-hop', or 'Qualifier'")
    question: str = Field(description="Natural language question in English")
    expected_focus: str = Field(description="Key information this question tests")

class QuestionList(BaseModel):
    questions: List[GeneratedQuestion]

class RAGEvaluation(BaseModel):
    original_score: float = Field(ge=0.0, le=10.0)
    hyper_score: float = Field(ge=0.0, le=10.0)
    winner: str = Field(description="'Original', 'Hyper', or 'Tie'")
    reason: str = Field(description="한국어로 점수 부여 및 승자 선정 이유")

# ─────────────────────────────────────────────
# 2. 엔티티 추출
# ─────────────────────────────────────────────
def extract_entities(question: str) -> List[str]:
    """질문에서 핵심 엔티티 이름을 추출합니다."""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"""Extract the key named entity names (people, organizations, places, works) 
from this question. Return only the entity names, no descriptions.

Question: {question}""",
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=ExtractedEntities,
            temperature=0,
        ),
    )
    return response.parsed.entities

# ─────────────────────────────────────────────
# 3. 서브그래프 조회 (고정 Cypher 패턴)
# ─────────────────────────────────────────────
def get_subgraph_orig(graph: Neo4jGraph, entities: List[str]) -> str:
    """
    Original DB: 기본 triple 구조
    노드: (:Entity {name}), 관계: [:REL_TYPE]
    """
    if not entities:
        return "No entities found."

    where_clauses = " OR ".join(
        [f"toLower(n.name) CONTAINS toLower('{e}')" for e in entities]
    )

    cypher = f"""
MATCH (n)-[r]->(m)
WHERE {where_clauses}
RETURN n.name AS head, labels(n) AS head_labels,
       type(r) AS relation,
       m.name AS tail, labels(m) AS tail_labels
LIMIT {MAX_SUBGRAPH_RESULTS}
"""
    try:
        results = graph.query(cypher)
        if not results:
            # 1-hop 이웃이 없으면 노드 자체만 반환
            cypher2 = f"""
MATCH (n)
WHERE {where_clauses}
RETURN n.name AS name, labels(n) AS labels
LIMIT 10
"""
            results = graph.query(cypher2)
            if not results:
                return "No relevant information found."
            return "Entities found:\n" + "\n".join(
                [f"- {r['name']} ({', '.join(r['labels'])})" for r in results]
            )

        lines = []
        for r in results:
            lines.append(
                f"({r['head']}) --[{r['relation']}]--> ({r['tail']})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Query error: {e}"


def get_subgraph_hyper(graph: Neo4jGraph, entities: List[str]) -> str:
    """
    Hyper DB: qualifier 프로퍼티 포함
    노드: (:Entity:TYPE {name}), 관계: [:REL_TYPE {WHO, WHAT, WHEN, WHERE, WHY, HOW, doc_id}]
    """
    if not entities:
        return "No entities found."

    where_clauses = " OR ".join(
        [f"toLower(n.name) CONTAINS toLower('{e}')" for e in entities]
    )

    # 1-hop + qualifier 프로퍼티 포함
    cypher = f"""
MATCH (n)-[r]->(m)
WHERE {where_clauses}
RETURN n.name AS head,
       [l IN labels(n) WHERE l <> 'Entity'] AS head_type,
       type(r) AS relation,
       m.name AS tail,
       [l IN labels(m) WHERE l <> 'Entity'] AS tail_type,
       r.WHO   AS who,
       r.WHAT  AS what,
       r.WHEN  AS when,
       r.WHERE AS where,
       r.WHY   AS why,
       r.HOW   AS how
LIMIT {MAX_SUBGRAPH_RESULTS}
"""
    try:
        results = graph.query(cypher)

        # 결과 없으면 2-hop 시도
        if not results:
            where_clauses2 = " OR ".join(
                [f"toLower(n.name) CONTAINS toLower('{e}')" for e in entities]
            )
            cypher2 = f"""
MATCH (n)-[r1]->(mid)-[r2]->(m)
WHERE {where_clauses2}
RETURN n.name AS head, type(r1) AS rel1,
       mid.name AS mid, type(r2) AS rel2,
       m.name AS tail,
       r1.WHEN AS when1, r2.WHEN AS when2
LIMIT 15
"""
            results = graph.query(cypher2)
            if not results:
                return "No relevant information found."

            lines = [f"(Multi-hop results)"]
            for r in results:
                lines.append(
                    f"({r['head']}) --[{r['rel1']}]--> ({r['mid']}) --[{r['rel2']}]--> ({r['tail']})"
                    + (f"  | WHEN: {r['when1']}" if r.get('when1') else "")
                )
            return "\n".join(lines)

        # qualifier 있는 결과 포맷팅
        lines = []
        for r in results:
            head_type = r['head_type'][0] if r['head_type'] else '?'
            tail_type = r['tail_type'][0] if r['tail_type'] else '?'
            base = f"({r['head']}:{head_type}) --[{r['relation']}]--> ({r['tail']}:{tail_type})"

            quals = []
            for key in ['who', 'what', 'when', 'where', 'why', 'how']:
                val = r.get(key)
                if val:
                    quals.append(f"{key.upper()}: {val}")

            if quals:
                base += f"  | {', '.join(quals)}"
            lines.append(base)

        return "\n".join(lines)

    except Exception as e:
        return f"Query error: {e}"


# ─────────────────────────────────────────────
# 4. GraphRAG 답변 생성
# ─────────────────────────────────────────────
def answer_with_graph(question: str, subgraph_context: str, db_name: str) -> str:
    """서브그래프 컨텍스트를 기반으로 LLM이 답변을 생성합니다."""
    prompt = f"""You are a Knowledge Graph QA system.
Answer the question using ONLY the information provided in the graph context below.
If the answer is not in the context, say "I don't have enough information."

[Graph Context from {db_name}]
{subgraph_context}

[Question]
{question}

Answer concisely and directly."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text.strip()
    except Exception as e:
        return f"답변 생성 실패: {e}"


# ─────────────────────────────────────────────
# 5. 질문 생성
# ─────────────────────────────────────────────
def get_doc_and_graph(doc_data):
    doc_text   = " ".join([" ".join(sent) for sent in doc_data.get('sents', [])])
    graph_info = json.dumps(doc_data.get('labels', []), ensure_ascii=False)
    return doc_text, graph_info

def generate_questions_for_doc(doc_text, graph_info):
    prompt = f"""You are a question designer for evaluating GraphRAG systems.
Generate 3 questions in English based on the document and its knowledge graph:

1. Simple: A direct 1-hop relationship question.
2. Multi-hop: Requires traversing A→B→C chain.
3. Qualifier: Requires reading a qualifier (WHO/WHAT/WHEN/WHERE/WHY/HOW) on an edge.

Each question must have a clear, verifiable answer in the graph.

Document: {doc_text}
Graph Info (qualifiers included): {graph_info}
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


def evaluate_rag_responses(question, expected_focus, ans_orig, ans_hyper):
    prompt = f"""당신은 AI QA 시스템의 성능을 평가하는 심사위원입니다.

[질문]: {question}
[채점 핵심 기준]: {expected_focus}

[Original DB 답변]
{ans_orig}

[Hyper DB 답변]
{ans_hyper}

두 답변을 비교하여 채점 기준을 얼마나 잘 충족했는지 평가하세요.
특히 Hyper DB가 qualifier 정보(시기, 이유, 방법 등)를 더 잘 활용했는지 확인하세요.
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
        print(f"평가 실패: {e}")
        return None


# ─────────────────────────────────────────────
# 6. 실행
# ─────────────────────────────────────────────
if __name__ == "__main__":
    with open(V4_DATASET_PATH, 'r', encoding='utf-8') as f:
        all_docs = [json.loads(line) for line in f]

    sample_docs = random.sample(all_docs, min(SAMPLE_DOCS_COUNT, len(all_docs)))

    print("\n📝 1단계: 평가 질문 생성 중...\n")
    test_questions = []
    for doc in sample_docs:
        doc_text, graph_info = get_doc_and_graph(doc)
        qs = generate_questions_for_doc(doc_text, graph_info)
        test_questions.extend(qs)

    print(f"총 {len(test_questions)}개 질문 생성 완료")
    for q in test_questions:
        print(f"  [{q.question_type}] {q.question}")

    print("\n🚀 2단계: GraphRAG 성능 비교 테스트 시작...\n")
    results = []

    for i, q_obj in enumerate(test_questions):
        question = q_obj.question
        print(f"\n[{i+1}/{len(test_questions)}] {question}  (유형: {q_obj.question_type})")

        # 엔티티 추출
        entities = extract_entities(question)
        print(f"  🔍 추출된 엔티티: {entities}")

        # 서브그래프 조회
        subgraph_orig  = get_subgraph_orig(graph_orig,   entities)
        subgraph_hyper = get_subgraph_hyper(graph_hyper, entities)

        # 답변 생성
        ans_orig  = answer_with_graph(question, subgraph_orig,  "Original DB")
        ans_hyper = answer_with_graph(question, subgraph_hyper, "Hyper DB")

        print(f"  └ Original : {ans_orig}")
        print(f"  └ Hyper    : {ans_hyper}")

        # 평가
        eval_result = evaluate_rag_responses(question, q_obj.expected_focus, ans_orig, ans_hyper)
        if eval_result:
            print(f"  🏆 Winner: {eval_result.winner}  (Orig {eval_result.original_score} vs Hyper {eval_result.hyper_score})")
            print(f"  💡 {eval_result.reason}")

            results.append({
                "Question_Type":    q_obj.question_type,
                "Question":         question,
                "Expected_Focus":   q_obj.expected_focus,
                "Entities":         ", ".join(entities),
                "Original_Context": subgraph_orig[:300],
                "Hyper_Context":    subgraph_hyper[:300],
                "Original_Answer":  ans_orig,
                "Hyper_Answer":     ans_hyper,
                "Original_Score":   eval_result.original_score,
                "Hyper_Score":      eval_result.hyper_score,
                "Winner":           eval_result.winner,
                "Reason":           eval_result.reason,
            })

    if results:
        df = pd.DataFrame(results)
        filename = f"graphrag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"\n✅ 완료! 결과 저장: {filename}")