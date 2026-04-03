import json
from pprint import pprint

# 사용자가 제공한 샘플 데이터
sample_data = {
    "tokens": ["She", "is", "known", "for", "her", "two", "best", "-", "selling", "novels", ",", "The", "Fountainhead", "(", "1943", ")", "and", "Atlas", "Shrugged", "(", "1957", ")", ",", "and", "for", "developing", "a", "philosophical", "system", "she", "called", "Objectivism", "."],
    "entities": [
        {"span": [0, 1], "label": "Entity"},
        {"span": [11, 13], "label": "Entity"},
        {"span": [14, 15], "label": "Entity"}
    ],
    "relations": [
        {
            "head": [0, 1],
            "tail": [11, 13],
            "label": "notable work",
            "qualifiers": [
                {"span": [14, 15], "label": "publication date"}
            ]
        }
    ]
}

def parse_hyperred_sample(data: dict) -> dict:
    tokens = data["tokens"]
    
    # 1. 토큰을 LLM에 넣을 자연어 텍스트로 복원
    # (주의: 영어의 경우 구두점 띄어쓰기가 완벽하진 않지만, LLM이 문맥을 파악하는 데는 충분합니다)
    raw_text = " ".join(tokens)
    
    # 스팬(Span) 인덱스를 기반으로 텍스트를 추출하는 헬퍼 함수
    def get_text_from_span(span: list) -> str:
        return " ".join(tokens[span[0]:span[1]])
    
    parsed_facts = []
    
    # 2. Relation (Base Triple + Qualifiers) 파싱
    for rel in data.get("relations", []):
        head_text = get_text_from_span(rel["head"])
        tail_text = get_text_from_span(rel["tail"])
        relation_label = rel["label"]
        
        # Qualifier 파싱
        parsed_qualifiers = []
        for qual in rel.get("qualifiers", []):
            parsed_qualifiers.append({
                "qualifier_relation": qual["label"],
                "qualifier_value": get_text_from_span(qual["span"])
            })
            
        parsed_facts.append({
            "head": head_text,
            "relation": relation_label,
            "tail": tail_text,
            "qualifiers": parsed_qualifiers
        })
        
    return {
        "raw_text": raw_text,
        "ground_truth_facts": parsed_facts
    }

# --- 실행 테스트 ---
if __name__ == "__main__":
    result = parse_hyperred_sample(sample_data)
    
    print("=== 📝 LLM Input Text (원문 복원) ===")
    print(result["raw_text"])
    print("\n=== 🎯 Ground Truth Facts (정답지 변환) ===")
    pprint(result["ground_truth_facts"], sort_dicts=False)