import os
import json
import re
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

# 1. 환경 변수 로드 및 Neo4j 연결
load_dotenv() 
print("🔌 Neo4j에 연결을 시도합니다...")
graph = Neo4jGraph()
print("✅ Neo4j 연결 성공!")

# 2. P-ID -> 관계명 맵핑
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
    "P1441": "present in work", "P3373": "sibling",
}

def format_rel_name(text):
    return re.sub(r'[^a-zA-Z0-9]', '_', text).upper()

# 3. 데이터 로드
DATASET_PATH = "./results/redocred_original.jsonl" 
DOC_LIMIT = 582

print(f"📖 데이터 파싱 시작... (처음 {DOC_LIMIT}개 문서 처리)")

nodes_dict = {} 
edges_list = []

with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    for doc_id, line in enumerate(f):
        if doc_id >= DOC_LIMIT: break
            
        data = json.loads(line)
        local_idx_to_name = {}
        
        # 노드 파싱
        for idx, v_group in enumerate(data.get('vertexSet', [])):
            if not v_group: continue
            entity = v_group[0]
            name = entity['name'].strip()
            local_idx_to_name[idx] = name
            
            # 데이터셋에 이미 정제된 type 값을 그대로 사용 (예외 방지용 기본값 지정)
            e_type = entity.get('type', 'MISC').strip()
            
            # 특수문자 제거 후 안전한 형태로 저장 (ex. "LOC" -> "LOC")
            safe_type = re.sub(r'[^a-zA-Z0-9]', '', e_type).upper()
            if not safe_type: safe_type = "MISC"
            
            nodes_dict[name] = {"name": name, "type": safe_type}
            
        # 엣지 파싱
        for label in data.get('labels', []):
            h_name = local_idx_to_name.get(label['h'])
            t_name = local_idx_to_name.get(label['t'])
            if not h_name or not t_name: continue
                
            raw_rel = RELATION_MAP.get(label['r'], label['r'])
            rel_type = format_rel_name(raw_rel)
            
            properties = {"doc_id": doc_id}
            for q in label.get('qualifiers', []):
                properties[q['qualifier']] = q['qualifier_value']
                
            edges_list.append({
                "head": h_name, "tail": t_name, "rel_type": rel_type, "props": properties
            })

print(f"✅ 파싱 완료: 총 {len(nodes_dict)}개의 노드와 {len(edges_list)}개의 엣지가 준비되었습니다.")

# 4. Neo4j에 노드 삽입
print("🏗️ 노드(엔티티) 생성 중... (데이터셋의 원본 Type 라벨 적용)")
nodes_by_type = {}
for node in nodes_dict.values():
    n_type = node["type"]
    if n_type not in nodes_by_type:
        nodes_by_type[n_type] = []
    nodes_by_type[n_type].append({"name": node["name"]})

for n_type, node_list in nodes_by_type.items():
    # n_type은 데이터셋에 적힌 ORG, LOC, PER, TIME, NUM, MISC 중 하나가 들어갑니다.
    node_query = f"""
    UNWIND $nodes AS n
    MERGE (e:Entity:{n_type} {{name: n.name}})
    """
    graph.query(node_query, params={"nodes": node_list})

# 5. Neo4j에 엣지 삽입
print("🔗 관계(엣지) 연결 중... (Qualifier 포함)")
for edge in edges_list:
    edge_query = f"""
    MATCH (h:Entity {{name: $head}})
    MATCH (t:Entity {{name: $tail}})
    MERGE (h)-[r:{edge['rel_type']}]->(t)
    SET r += $props 
    """
    graph.query(edge_query, params={"head": edge["head"], "tail": edge["tail"], "props": edge["props"]})

print("🎉 완벽합니다! 원본 데이터셋의 카테고리가 그대로 반영된 지식 그래프가 성공적으로 구축되었습니다!")