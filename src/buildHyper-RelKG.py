import json
import os
from pyvis.network import Network

# 1. Wikidata P-ID 매핑 딕셔너리 (필요한 것들을 계속 추가하세요!)
RELATION_MAP = {
    "P17": "country", "P31": "instance of", "P131": "located in", 
    "P159": "headquarters location", "P355": "subsidiary", 
    "P571": "inception", "P576": "dissolved date", "P585": "point in time",
    "P749": "parent organization", "P1056": "product produced",
    "P150": "contains administrative territorial entity"
}

def create_interactive_kg(data_path, output_html="knowledge_graph.html"):
    # 다크 테마 설정
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.force_atlas_2based()

    if not os.path.exists(data_path):
        print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line: return
        doc = json.loads(line)
        
    print(f"[{doc.get('title', 'Unknown')}] 문서의 지식 그래프를 변환 및 시각화 중...")

    for h_lbl in doc.get('hyper_labels', []):
        if not h_lbl or 'fact_id' not in h_lbl: continue
        
        # 2. 원본 Relation ID를 텍스트로 변환
        raw_rel = h_lbl.get('refined_relation', 'related to')
        # RELATION_MAP에 있으면 변환, 없으면 원래 ID 유지 (예: P123)
        rel_text = RELATION_MAP.get(raw_rel, raw_rel)

        # 주어(Head) 이름 가져오기
        fact_id = h_lbl['fact_id']
        try:
            # 원본 labels에서 head 인덱스 참조
            h_idx = doc['labels'][fact_id]['h']
            h_name = doc['vertexSet'][h_idx][0]['name']
        except (IndexError, KeyError):
            continue

        t_name = h_lbl.get('refined_tail')
        
        # 3. 툴팁(Tooltip) 구성: 변환된 관계명과 Qualifier 표시
        qualifiers = h_lbl.get('qualifiers', [])
        tooltip_text = f"Relation ID: {raw_rel}\nMeaning: {rel_text}\n"
        for q in qualifiers:
            tooltip_text += f" - {q['qualifier_relation']}: {q['qualifier_value_text']}\n"

        # 노드 추가 (색상 구분)
        net.add_node(h_name, label=h_name, title="Entity", color="#00ffcc")
        
        if t_name:
            net.add_node(t_name, label=t_name, title="Entity", color="#ff99cc")
            # 엣지 추가 (label에는 사람이 읽는 텍스트, title에는 마우스 오버 상세 정보)
            net.add_edge(source=h_name, to=t_name, label=rel_text, title=tooltip_text, color="#aaaaaa")

    # 저장 및 실행
    net.show(output_html, notebook=False)
    print(f"✅ 완성! {os.path.abspath(output_html)} 파일을 브라우저로 확인하세요.")

if __name__ == "__main__":
    # 파일 경로가 맞는지 다시 한번 확인해주세요!
    DATA_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v2.jsonl"
    create_interactive_kg(DATA_PATH)