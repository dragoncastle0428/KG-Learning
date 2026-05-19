import json
import os

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

VALID_QUALIFIER_TYPES = {"WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"}

QUALIFIER_REMAP = {
    "ROLE":     "HOW",
    "POSITION": "HOW",
    "AS":       "HOW",
    "MANNER":   "HOW",
    "METHOD":   "HOW",
    "RELATION": "WHO",
    "PERSON":   "WHO",
    "AGENT":    "WHO",
    "CAUSE":    "WHY",
    "REASON":   "WHY",
    "PURPOSE":  "WHY",
    "TIME":     "WHEN",
    "DATE":     "WHEN",
    "PERIOD":   "WHEN",
    "PLACE":    "WHERE",
    "LOCATION": "WHERE",
    "REGION":   "WHERE",
    "TOPIC":    "WHAT",
    "TYPE":     "WHAT",
    "NATURE":   "WHAT",
    "ORG":      None,
    "PER":      None,
    "LOC":      None,
    "NUM":      None,
    "MISC":     None,
}

VAGUE_QUALIFIER_VALUES = {
    "currently", "now", "today", "current",
    "formerly", "previously", "originally", "initially",
    "later", "former", "then", "also", "afterward",
    "newly", "briefly", "recently",
}

# --- Qualifier 정규화 ---

def normalize_qualifier(q: dict) -> dict | None:
    qtype = q["qualifier"].strip().upper()
    if qtype in VALID_QUALIFIER_TYPES:
        return q
    if qtype in QUALIFIER_REMAP:
        mapped = QUALIFIER_REMAP[qtype]
        if mapped is None:
            return None
        return {"qualifier": mapped, "qualifier_value": q["qualifier_value"]}
    return None

def filter_qualifiers(qualifiers: list) -> list:
    result = []
    for q in qualifiers:
        q = normalize_qualifier(q)
        if q is None:
            continue
        if q["qualifier_value"].strip().lower() in VAGUE_QUALIFIER_VALUES:
            continue
        result.append(q)
    return result

# --- 메인 후처리 함수 ---

def postprocess_document(sample: dict) -> dict:
    labels = sample["labels"]

    # 유효하지 않은 P-ID 엣지 제거 + qualifier 정제
    filtered_labels = []
    for lbl in labels:
        if lbl["r"] not in RELATION_MAP:
            continue
        lbl = dict(lbl)
        lbl["qualifiers"] = filter_qualifiers(lbl.get("qualifiers", []))
        filtered_labels.append(lbl)

    sample = dict(sample)
    sample["labels"] = filtered_labels
    return sample

# --- 통계 ---

def compute_stats(before: dict, after: dict) -> dict:
    before_quals = {(l["h"], l["r"], l["t"]): l.get("qualifiers", []) for l in before["labels"]}
    remapped = 0
    for lbl in after["labels"]:
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key in before_quals:
            for q_after in lbl.get("qualifiers", []):
                for q_before in before_quals[key]:
                    if q_before["qualifier_value"] == q_after["qualifier_value"] and q_before["qualifier"] != q_after["qualifier"]:
                        remapped += 1
    return {
        "edges_removed": len(before["labels"]) - len(after["labels"]),
        "qualifiers_removed": sum(len(l.get("qualifiers", [])) for l in before["labels"])
                            - sum(len(l.get("qualifiers", [])) for l in after["labels"]),
        "qualifiers_remapped": remapped,
    }

# --- 실행 ---

def run_postprocess(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_edges_removed = 0
    total_qualifiers_removed = 0
    total_qualifiers_remapped = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(documents):
            processed = postprocess_document(sample)
            stats = compute_stats(sample, processed)

            total_edges_removed += stats["edges_removed"]
            total_qualifiers_removed += stats["qualifiers_removed"]
            total_qualifiers_remapped += stats["qualifiers_remapped"]

            f.write(json.dumps(processed, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(documents)}] 처리 중...")

    print(f"\n✅ 후처리 완료: {len(documents)}개 문서")
    print(f"   엣지 제거:      {total_edges_removed}개")
    print(f"   Qualifier 제거: {total_qualifiers_removed}개")
    print(f"   Qualifier 재매핑: {total_qualifiers_remapped}개")
    
    
if __name__ == "__main__":
    INPUT_FILE  = "./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl"
    OUTPUT_FILE = "./Dataset/HypeRelDocRED/train/train_enriched_v4_clean.jsonl"

    run_postprocess(INPUT_FILE, OUTPUT_FILE)