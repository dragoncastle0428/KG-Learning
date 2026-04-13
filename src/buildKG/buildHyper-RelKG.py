import json
import os
from pyvis.network import Network

# 1. Wikidata P-ID 매핑 딕셔너리
# 1. Wikidata P-ID 매핑 딕셔너리 (최종_진짜최종 업데이트)
# 1. Wikidata P-ID 매핑 딕셔너리 (전수 검증 완료 버전)
RELATION_MAP = {
    # 🏢 [조직/기업 관련 (ORG)]
    "P571": "inception / founded in",         # 설립일/창립일
    "P112": "founded by",                     # 창립자
    "P159": "headquarters location",          # 본사/중심지 위치
    "P127": "owned by",                       # 소유주 (모회사/지주회사)
    "P749": "parent organization",            # 상위 조직
    "P355": "subsidiary",                     # 자회사/하위 조직
    "P155": "follows / preceded by",          # 이전 이름/전신 
    "P156": "followed by",                    # 이후 이름/후신
    "P1365": "replaces",                      # ~를 대체함/인수함
    "P1366": "replaced by",                   # ~로 대체됨/인수합병됨
    "P1416": "affiliation / strategic alliance", # 💡수정됨 (제휴/협력/관계사)
    "P452": "industry",                       # 소속 산업군
    "P7888": "merged into",
    "P121": "operated",

    # 👤 [인물 관련 (PER)]
    "P569": "date of birth",                  # 출생일
    "P570": "date of death",                  # 사망일
    "P19": "place of birth",                  # 출생지
    "P20": "place of death",                  # 사망지
    "P106": "occupation",                     # 직업
    "P108": "employer",                       # 고용주/소속 직장
    "P69": "educated at",                     # 출신 학교/학력
    "P26": "spouse",                          # 배우자
    "P22": "father",                          # 아버지
    "P25": "mother",                          # 어머니

    # 🌍 [지리/장소 관련 (LOC)]
    "P17": "country",                         # 소속 국가
    "P131": "located in",                     # 위치한 행정 구역 (도시, 주 등)
    "P36": "capital",                         # 수도
    "P47": "shares border with",              # 인접 지역/국경을 맞댐
    "P1992": "proximate points of interest", 

    # 🎬 [사건/창작물/기타 (TIME, EVENT, WORK)]
    "P585": "point in time",                  # 특정 사건이 일어난 시점
    "P577": "publication date",               # 출판/발표일
    "P50": "author",                          # 저자
    "P57": "director",                        # 감독
    "P175": "performer",                      # 연주자/출연자
    "P166": "award received",                 # 수상 내역
    "P1411": "nominated for",                 # 후보 지명
    
    # 💡 [특수 도메인 (항공사 등 문서 맞춤형)]
    "P1056": "service provided",              # 💡수정됨 (서비스/제품 제공)
    "P1444": "destination point",             # 💡수정됨 (여정/항공편의 목적지)
    "P3608": "regulated by"                   # 💡수정됨 (규제/관리 감독 기관)
}

# 2. 노드 타입별 색상 매핑 (다크 테마에 어울리는 파스텔톤)
COLOR_MAP = {
    "ORG": "#66b3ff",      # 파란색 (조직)
    "LOC": "#99ff99",      # 초록색 (장소)
    "CONCEPT": "#ffcc99",  # 주황색 (개념/서비스)
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
    DATA_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v3.jsonl"
    
    # 임시 테스트를 원하시면, 위에서 주신 JSON을 파일로 저장 후 돌려보세요!
    create_interactive_kg(DATA_PATH, output_html="airasia_zest_kg.html")