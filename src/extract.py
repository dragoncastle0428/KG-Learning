import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List

# 1. 환경 변수 로드 (.env 파일 안의 내용을 불러옴)
# 이 코드가 실행되면 os.getenv()로 API 키를 안전하게 가져올 수 있습니다.
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다!")

# Gemini API 클라이언트 설정
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. Pydantic 스키마 정의 (이전과 동일) ---
class Entity(BaseModel):
    entity_id: str = Field(description="Unique name or value of the entity (e.g., 'OpenAI', 'San Francisco')")
    entity_type: str = Field(description="Broad category of the entity (e.g., 'Organization', 'Location', 'Time', 'Product')")

class Event(BaseModel):
    event_id: str = Field(description="Unique identifier for the event (e.g., 'Event_Launch_01')")
    event_type: str = Field(description="Category of the event or action (e.g., 'Product_Launch')")
    description: str = Field(description="A brief, one-sentence summary of what happened")

class RoleEdge(BaseModel):
    entity_id: str = Field(description="The ID of the participating entity")
    event_id: str = Field(description="The ID of the central event")
    role: str = Field(description="The specific role the entity plays (e.g., 'Subject', 'Object', 'Location', 'Time')")

class HypergraphKG(BaseModel):
    entities: List[Entity] = Field(description="List of all extracted entities")
    events: List[Event] = Field(description="List of all extracted events")
    edges: List[RoleEdge] = Field(description="List of edges connecting entities to events")

# --- 3. Gemini 추출 함수 ---
def extract_hypergraph_with_gemini(text: str) -> HypergraphKG:
    # 성능과 속도의 균형이 좋은 flash 모델 사용 (복잡한 논문은 pro 모델 권장)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    system_instruction = """
    You are an expert in Knowledge Graph extraction, specifically focusing on Hypergraph structures using Reification.
    Analyze the provided text and model relationships as central 'Event' nodes.
    Extract all participants, times, locations, and methods as independent 'Entity' nodes.
    Connect each Entity to the Event using a 'RoleEdge', clearly defining their role.
    """
    
    prompt = f"{system_instruction}\n\n[Text to Analyze]\n{text}"

    # Gemini에게 Pydantic 스키마를 전달하여 JSON 형식으로만 답변하도록 강제
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=HypergraphKG,
        ),
    )
    
    # 반환된 JSON 문자열을 다시 Pydantic 객체로 변환하여 안전하게 사용
    parsed_data = HypergraphKG.model_validate_json(response.text)
    return parsed_data

# --- 4. 실행 테스트 ---
if __name__ == "__main__":
    sample_text = """
    On November 30, 2022, OpenAI officially launched ChatGPT, which is based on a large language model, 
    in San Francisco to assist people with their daily tasks.
    """
    
    print("Gemini 모델로 하이퍼그래프 KG 추출 중...\n")
    try:
        hg_result = extract_hypergraph_with_gemini(sample_text)
        
        print("=== 🟢 Entities (Nodes) ===")
        for e in hg_result.entities:
            print(f" - [{e.entity_type}] {e.entity_id}")
            
        print("\n=== 🔴 Events (Hyperedges) ===")
        for ev in hg_result.events:
            print(f" - {ev.event_id} ({ev.event_type}): {ev.description}")
            
        print("\n=== 🔗 Role Connections (Edges) ===")
        for edge in hg_result.edges:
            print(f" [{edge.entity_id}] --({edge.role})--> [{edge.event_id}]")
            
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")