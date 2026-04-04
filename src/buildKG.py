import os
from datasets import Dataset, load_from_disk
from pyvis.network import Network

# 보기 편하도록 자주 쓰이는 위키데이터 속성(P-ID)을 매핑합니다.
RELATION_MAP = {
    "P159": "headquarters location", "P17": "country", 
    "P131": "located in", "P150": "contains", 
    "P576": "dissolved date", "P355": "subsidiary", 
    "P749": "parent organization"
}

def create_redocred_kg(orig_path, target_index=0, output_html="redocred_baseline_kg.html"):
    # 1. 다크 테마 그래프 세팅
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.force_atlas_2based()

    # 2. 데이터 로드 (Hugging Face 형식)
    print("⏳ 원본 데이터를 불러오는 중입니다...")
    try:
        if os.path.isdir(orig_path):
            orig_dataset = load_from_disk(orig_path)
            orig_doc = orig_dataset['train'][target_index] if 'train' in orig_dataset else orig_dataset[target_index]
        elif orig_path.endswith('.arrow'):
            orig_dataset = Dataset.from_file(orig_path)
            orig_doc = orig_dataset[target_index]
        else:
            raise ValueError("지원하지 않는 원본 데이터 형식입니다.")
    except Exception as e:
        print(f"❌ 원본 데이터를 불러오는 데 실패했습니다: {e}")
        return

    print(f"[{orig_doc.get('title', 'Unknown')}] 원본 문서의 지식 그래프를 그립니다...")

    # 3. 원본 정적 팩트(labels)를 노드와 엣지로 변환
    # Re-DocRED의 구조: Head(h) -- Relation(r) --> Tail(t)
    for lbl in orig_doc.get('labels', []):
        try:
            # 주어(Head)와 목적어(Tail) 이름 가져오기
            h_name = orig_doc['vertexSet'][lbl['h']][0]['name']
            t_name = orig_doc['vertexSet'][lbl['t']][0]['name']
        except (IndexError, KeyError):
            continue
            
        # P-ID (예: P576)를 알아보기 쉬운 영어 단어로 변환
        r_id = lbl['r']
        r_name = RELATION_MAP.get(r_id, r_id) 

        # 노드 추가 (일반 엔티티는 민트색, 꼬리 노드는 핑크색)
        net.add_node(h_name, label=h_name, title="Entity", color="#00ffcc")
        net.add_node(t_name, label=t_name, title="Entity", color="#ff99cc")
        
        # 엣지(선) 추가
        net.add_edge(source=h_name, to=t_name, label=r_name, title=f"Relation: {r_name}", color="#aaaaaa")

    # 4. HTML 파일로 저장
    net.show(output_html, notebook=False)
    print(f"✅ 완성! {output_html} 파일을 열어보세요.")

if __name__ == "__main__":
    # 🌟 사용자님의 로컬 Re-DocRED 데이터 경로로 수정해주세요!
    ORIGINAL_PATH = "./Dataset/Re-DocRED/train" 
    
    # AirAsia Zest(0번) 문서 기준으로 그래프 생성
    create_redocred_kg(ORIGINAL_PATH, target_index=0)