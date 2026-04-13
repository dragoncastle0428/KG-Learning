import os
import json
from collections import Counter
from datasets import load_from_disk
import matplotlib.pyplot as plt

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic' # Mac은 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 환경 설정 및 RELATION_MAP ---
ORIGINAL_DATASET_PATH = "./Dataset/Re-DocRED"
NEW_DATASET_PATH = "./Dataset/HypeRelDocRED/train/train_enriched_v3.jsonl"
COMPARE_LIMIT = 582 # 비교할 문서 개수 제한

RELATION_MAP = {
    # 🏢 [조직/기업 관련 (ORG)]
    "P571": "founded in", 
    "P112": "founded by", 
    "P159": "headquarters", 
    "P127": "owned by", 
    "P749": "parent org", 
    "P355": "subsidiary", 
    "P155": "preceded by", 
    "P156": "followed by", 
    "P1365": "replaces", 
    "P1366": "replaced by", 
    "P1416": "affiliation", 
    "P452": "industry", 
    "P7888": "merged into",
    "P121": "operated",
    # 👤 [인물 관련 (PER)]
    "P569": "date of birth", 
    "P570": "date of death", 
    "P19": "place of birth", 
    "P20": "place of death", 
    "P106": "occupation", 
    "P108": "employer", 
    "P69": "educated at", 
    "P26": "spouse", 
    "P22": "father", 
    "P25": "mother", 
    # 🌍 [지리/장소 관련 (LOC)]
    "P17": "country", 
    "P131": "located in", 
    "P36": "capital", 
    "P47": "shares border", 
    "P1992": "proximate points", 
    # 🎬 [사건/창작물/기타 (TIME, EVENT, WORK)]
    "P585": "point in time", 
    "P577": "publication date", 
    "P50": "author", 
    "P57": "director", 
    "P175": "performer", 
    "P166": "award received", 
    "P1411": "nominated for", 
    "P1056": "service provided", 
    "P1444": "destination point", 
    "P3608": "regulated by" 
}

def load_new_dataset(jsonl_path):
    data = []
    if not os.path.exists(jsonl_path):
        print(f"⚠️ 파일을 찾을 수 없습니다: {jsonl_path}")
        return data
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_datasets():
    print("데이터를 불러오는 중입니다...\n")
    
    try:
        original_ds = load_from_disk(ORIGINAL_DATASET_PATH)["train"]
        # 지정된 개수(743개)만큼만 슬라이싱
        original_data = [doc for doc in original_ds][:COMPARE_LIMIT]
    except Exception as e:
        print(f"원본 데이터 로드 실패: {e}")
        return

    # 새 데이터도 동일하게 슬라이싱
    new_data = load_new_dataset(NEW_DATASET_PATH)[:COMPARE_LIMIT]
    
    if not original_data or not new_data:
        return

    # --- 2. 통계 지표 계산 ---
    stats = {
        "Original": {"docs": len(original_data), "entities": 0, "relations": 0, "rel_counter": Counter()},
        "New": {"docs": len(new_data), "entities": 0, "relations": 0, "rel_counter": Counter(), "mapping_fails": 0}
    }

    for doc in original_data:
        stats["Original"]["entities"] += len(doc.get("vertexSet", []))
        stats["Original"]["relations"] += len(doc.get("labels", []))
        for label in doc.get("labels", []):
            stats["Original"]["rel_counter"][label["r"]] += 1

    for doc in new_data:
        vertex_set = doc.get("vertexSet", [])
        stats["New"]["entities"] += len(vertex_set)
        stats["New"]["relations"] += len(doc.get("labels", []))
        
        for v_group in vertex_set:
            if v_group[0].get("pos") == [-1, -1]:
                stats["New"]["mapping_fails"] += 1
                
        for label in doc.get("labels", []):
            stats["New"]["rel_counter"][label["r"]] += 1

    # --- 3. 콘솔 결과 출력 ---
    print("="*60)
    print(f" 📈 데이터셋 통계 비교 리포트 (첫 {COMPARE_LIMIT}개 문서 기준)")
    print("="*60)
    
    for name in ["Original", "New"]:
        docs = stats[name]["docs"]
        avg_ent = stats[name]["entities"] / docs if docs else 0
        avg_rel = stats[name]["relations"] / docs if docs else 0
        
        print(f"[{name} Dataset]")
        print(f" - 문서 수: {docs} 개")
        print(f" - 평균 노드 수: {avg_ent:.2f} 개 / 평균 엣지 수: {avg_rel:.2f} 개")
        
        if name == "New":
            fail_rate = (stats["New"]["mapping_fails"] / stats["New"]["entities"]) * 100 if stats["New"]["entities"] else 0
            print(f" - 🚨 좌표 매핑 실패 노드: {stats['New']['mapping_fails']} 개 ({fail_rate:.2f}%)")
        
        top3 = stats[name]["rel_counter"].most_common(3)
        top3_str = ", ".join([f"{RELATION_MAP.get(k, k)}({v}회)" for k, v in top3])
        print(f" - Top 3 관계: {top3_str}")
        print("-" * 60)

    # --- 4. 시각화 ---
    # 그래프 1: 평균 노드 및 엣지 수 비교
    plot_averages(stats)
    
    # 그래프 2: 관계(Relation) 분포 비교 (출력 개수를 25개로 확장)
    plot_relation_distribution(stats["Original"]["rel_counter"], stats["New"]["rel_counter"], top_n=25)


def plot_averages(stats):
    """평균 노드(Entities)와 엣지(Relations) 수를 막대 그래프로 비교합니다."""
    labels = ['Average Nodes (Entities)', 'Average Edges (Relations)']
    
    orig_docs = stats["Original"]["docs"] or 1
    new_docs = stats["New"]["docs"] or 1
    
    orig_means = [
        stats["Original"]["entities"] / orig_docs,
        stats["Original"]["relations"] / orig_docs
    ]
    new_means = [
        stats["New"]["entities"] / new_docs,
        stats["New"]["relations"] / new_docs
    ]
    
    x = range(len(labels))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar([i - width/2 for i in x], orig_means, width, label='Re-DocRED (Original)', color='#aab7b8')
    bars2 = plt.bar([i + width/2 for i in x], new_means, width, label='HypeRelDocRED (New)', color='#2980b9')
    
    plt.ylabel('Average Count per Document', fontsize=12)
    plt.title(f'Average Nodes and Edges per Document (n={stats["Original"]["docs"]})', fontsize=14, fontweight='bold')
    plt.xticks(x, labels, fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 막대 위에 정확한 수치 텍스트 추가
    for bar in bars1 + bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.show()


def plot_relation_distribution(orig_counter, new_counter, top_n=25):
    """가장 많이 등장한 관계들의 분포를 비교합니다. 출력 개수를 늘려 더 넓게 봅니다."""
    all_pids = set(orig_counter.keys()).union(set(new_counter.keys()))
    sorted_pids = sorted(list(all_pids), key=lambda x: new_counter.get(x, 0), reverse=True)[:top_n]
    
    orig_vals = [orig_counter.get(pid, 0) for pid in sorted_pids]
    new_vals = [new_counter.get(pid, 0) for pid in sorted_pids]
    
    x_labels = [f"{pid}\n({RELATION_MAP.get(pid, 'Unknown').split('/')[0].strip()})" for pid in sorted_pids]

    x = range(len(sorted_pids))
    width = 0.35

    # x축 항목이 많아졌으므로 그래프의 가로 길이를 16으로 더 넓게 설정
    plt.figure(figsize=(16, 7))
    plt.bar([i - width/2 for i in x], orig_vals, width, label='Re-DocRED (Original)', color='#aab7b8')
    plt.bar([i + width/2 for i in x], new_vals, width, label='HypeRelDocRED (New)', color='#2980b9')

    plt.xlabel('Relations (P-ID)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Top {top_n} Relation Distribution Comparison', fontsize=15, fontweight='bold')
    plt.xticks(x, x_labels, rotation=45, ha='right', fontsize=9)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_datasets()