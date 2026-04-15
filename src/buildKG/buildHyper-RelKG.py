import json
import os
from pyvis.network import Network

# 1. Wikidata P-ID 매핑 딕셔너리
# 1. Wikidata P-ID 매핑 딕셔너리 (최종_진짜최종 업데이트)
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
    "P1441": "present in work", "P3373": "sibling", "P190": "sister city"
}

# 2. 노드 타입별 색상 매핑 (다크 테마에 어울리는 파스텔톤)
COLOR_MAP = {
    "ORG": "#66b3ff",      # 파란색 (조직)
    "LOC": "#99ff99",      # 초록색 (장소)
    "MISC": "#ffcc99",  # 주황색 (개념/서비스)
    "PER": "#ff99cc",      # 핑크색 (인물)
    "TIME": "#ffff99",     # 노란색 (시간/날짜)
    "NUM": "#c2c2f0"       # 보라색 (숫자)
}

def create_interactive_kg(data_path, output_html="knowledge_graph.html"):
    # 다크 테마 설정
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    # 물리 엔진 설정 (노드들이 예쁘게 퍼지도록)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100)

    if not os.path.exists(data_path):
        print(f"❌ 파일을 찾을 수 없습니다: {data_path}")
        return

    # 첫 번째 줄(하나의 문서)만 읽어서 시각화
    with open(data_path, "r", encoding="utf-8") as f:
        line = f.readline()
        if not line: return
        doc = json.loads(line)
        
    print(f"[{doc.get('title', 'Unknown')}] 문서의 지식 그래프 시각화 중...")

    vertex_set = doc.get('vertexSet', [])
    labels = doc.get('labels', [])

    # --- STEP 1: 노드(Vertex) 생성 ---
    # vertexSet의 인덱스 번호가 고유 ID가 됩니다.
    for i, entity_group in enumerate(vertex_set):
        if not entity_group: continue
        
        node_data = entity_group[0]
        node_name = node_data['name']
        node_type = node_data.get('type', 'CONCEPT')
        
        # 타입에 맞는 색상 가져오기 (없으면 회색)
        node_color = COLOR_MAP.get(node_type, "#cccccc")
        
        # 마우스 오버 시 보일 툴팁
        tooltip = f"Type: {node_type}\nName: {node_name}"
        
        # 노드 추가 (id는 인덱스 숫자, 화면에 보이는 라벨은 텍스트)
        net.add_node(i, label=node_name, title=tooltip, color=node_color)

    # --- STEP 2: 엣지(Labels/Relations) 생성 ---
    for lbl in labels:
        h_idx = lbl['h']
        t_idx = lbl['t']
        raw_rel = lbl['r']
        
        # 원본 Relation ID를 텍스트로 변환 (MAP에 없으면 원본 텍스트 사용)
        rel_text = RELATION_MAP.get(raw_rel, raw_rel)

        # 툴팁(Tooltip) 구성: 변환된 관계명과 Qualifiers 표시
        qualifiers = lbl.get('qualifiers', [])
        tooltip_text = f"Relation: {rel_text}"
        if raw_rel != rel_text: # P-ID가 변환된 경우 원본 ID도 표시
            tooltip_text += f" ({raw_rel})"
            
        if qualifiers:
            tooltip_text += "\n" + "-"*20 + "\n"
            for q in qualifiers:
                # qualifier 구조 호환 (기존 qualifier_relation -> 변경된 qualifier)
                q_key = q.get('qualifier', q.get('qualifier_relation', 'INFO'))
                q_val = q.get('qualifier_value', q.get('qualifier_value_text', ''))
                tooltip_text += f"[{q_key}]: {q_val}\n"

        # 엣지 추가 (source와 to에 인덱스 번호 사용)
        net.add_edge(source=h_idx, to=t_idx, label=rel_text, title=tooltip_text, color="#aaaaaa")

    # 저장 및 실행
    net.show(output_html, notebook=False)
    print(f"✅ 완성! {os.path.abspath(output_html)} 파일을 브라우저로 열어보세요.")

if __name__ == "__main__":
    # 데이터셋 경로 지정 (위에서 제공해주신 JSON 구조가 들어있는 파일)
    DATA_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl"
    
    # 임시 테스트를 원하시면, 위에서 주신 JSON을 파일로 저장 후 돌려보세요!
    create_interactive_kg(DATA_PATH, output_html="airasia_zest_kg.html")