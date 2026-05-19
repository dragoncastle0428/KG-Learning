import os
import json
import time
from dotenv import load_dotenv
from google.genai import Client
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

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

load_dotenv()
API_KEYS = [os.getenv(f"GEMINI_API_KEY{i}") for i in range(1, 31) if os.getenv(f"GEMINI_API_KEY{i}")]
current_key_idx = 0

def get_current_client():
    global current_key_idx
    return Client(api_key=API_KEYS[current_key_idx])

# --- 스키마 ---
QualifierType = Literal["WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"]

class Qualifier(BaseModel):
    qualifier: QualifierType
    qualifier_value: str = Field(description="Contextual detail (within 5 words)")

class CorrectedLabel(BaseModel):
    h: str = Field(description="Exact head entity name from existing triples (for matching)")
    r: str = Field(description="Original P-ID (for matching)")
    t: str = Field(description="Exact tail entity name from existing triples (for matching)")
    new_r: str = Field(description="Corrected P-ID. Same as r if no change needed.")
    qualifiers: List[Qualifier] = Field(description="Corrected full qualifier list for this edge")

class MergeVertex(BaseModel):
    keep: str = Field(description="Entity name to keep")
    remove: str = Field(description="Entity name to remove and merge into keep")

class RemoveLabel(BaseModel):
    h: str = Field(description="Head entity name of the edge to remove")
    r: str = Field(description="P-ID of the edge to remove")
    t: str = Field(description="Tail entity name of the edge to remove")

class CorrectionResult(BaseModel):
    corrected_labels: List[CorrectedLabel]
    merge_vertices: List[MergeVertex]
    remove_labels: List[RemoveLabel]

# --- 유틸 ---
def flatten_document(sents: list) -> str:
    return " ".join([" ".join(sent) for sent in sents])

def format_existing_triples(vertex_set: list, labels: list) -> str:
    lines = []
    for lbl in labels:
        h_name = vertex_set[lbl['h']][0]['name']
        t_name = vertex_set[lbl['t']][0]['name']
        r_desc = RELATION_MAP.get(lbl['r'], lbl['r'])
        qualifiers = lbl.get('qualifiers', [])
        if qualifiers:
            qual_str = ", ".join([f"{q['qualifier']}: {q['qualifier_value']}" for q in qualifiers])
        else:
            qual_str = "none"
        lines.append(f"  ({h_name}) --[{lbl['r']}: {r_desc}]--> ({t_name}) | qualifiers: [{qual_str}]")
    return "\n".join(lines)

# --- 수정 사항 적용 ---
def apply_corrections(sample: dict, result: CorrectionResult) -> tuple[dict, dict]:
    vertex_set = [list(g) for g in sample["vertexSet"]]
    labels = [dict(l) for l in sample["labels"]]

    stats = {"merged": 0, "removed": 0, "r_changed": 0, "q_changed": 0}

    # 1. 버텍스 병합: remove → keep 으로 인덱스 재매핑
    name_to_idx = {vg[0]["name"].strip().lower(): i for i, vg in enumerate(vertex_set)}
    remap = {}  # old_idx → new_idx

    for mv in result.merge_vertices:
        keep_name = mv.keep.strip().lower()
        remove_name = mv.remove.strip().lower()
        if keep_name in name_to_idx and remove_name in name_to_idx:
            keep_idx = name_to_idx[keep_name]
            remove_idx = name_to_idx[remove_name]
            remap[remove_idx] = keep_idx
            stats["merged"] += 1

    # 버텍스 인덱스 재매핑 적용
    def resolve(idx):
        while idx in remap:
            idx = remap[idx]
        return idx

    for lbl in labels:
        lbl["h"] = resolve(lbl["h"])
        lbl["t"] = resolve(lbl["t"])

    # 2. 중복 엣지 제거 (remove_labels) — qualifier가 다르면 병합
    remove_set = set()
    for rl in result.remove_labels:
        h_name = rl.h.strip().lower()
        t_name = rl.t.strip().lower()
        if h_name in name_to_idx and t_name in name_to_idx:
            h_idx = resolve(name_to_idx[h_name])
            t_idx = resolve(name_to_idx[t_name])
            remove_set.add((h_idx, rl.r, t_idx))

    # 같은 (h,r,t)인데 qualifier가 다른 경우 → 삭제 대신 qualifier 병합
    # remove_labels에 지정된 엣지의 qualifier를 유지 엣지에 병합 후 제거
    keep_map = {}   # key → lbl (유지할 엣지)
    merge_q = {}    # key → 추가로 병합할 qualifier 목록
    for lbl in labels:
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key in remove_set:
            # 삭제 대상이지만 qualifier를 유지 엣지에 병합
            merge_q.setdefault(key, []).extend(lbl.get("qualifiers", []))
        else:
            keep_map[key] = lbl

    # 병합 적용
    for key, extra_quals in merge_q.items():
        if key in keep_map:
            existing = keep_map[key]
            seen_q = {(q["qualifier"], q["qualifier_value"]) for q in existing.get("qualifiers", [])}
            for q in extra_quals:
                qk = (q["qualifier"], q["qualifier_value"])
                if qk not in seen_q:
                    existing.setdefault("qualifiers", []).append(q)
                    seen_q.add(qk)
            stats["removed"] += 1
        # keep_map에 없으면 (유지 엣지 자체가 없는 경우) 그냥 제거
        else:
            stats["removed"] += 1

    labels = list(keep_map.values())

    # 3. 엣지 교정 (corrected_labels)
    correction_map = {}
    for cl in result.corrected_labels:
        h_name = cl.h.strip().lower()
        t_name = cl.t.strip().lower()
        if h_name in name_to_idx and t_name in name_to_idx:
            h_idx = resolve(name_to_idx[h_name])
            t_idx = resolve(name_to_idx[t_name])
            correction_map[(h_idx, cl.r, t_idx)] = cl

    for lbl in labels:
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key in correction_map:
            cl = correction_map[key]
            if cl.new_r != lbl["r"] and cl.new_r in RELATION_MAP:
                lbl["r"] = cl.new_r
                stats["r_changed"] += 1
            old_q = lbl.get("qualifiers", [])
            new_q = [q.model_dump() for q in cl.qualifiers]
            if old_q != new_q:
                lbl["qualifiers"] = new_q
                stats["q_changed"] += 1

    # 4. h == t인 셀프 루프 제거
    labels = [l for l in labels if l["h"] != l["t"]]

    # 5. 중복 엣지 최종 병합 (버텍스 병합 후 생긴 중복 — qualifier 합치기)
    seen = {}   # key → index in deduped
    deduped = []
    for lbl in labels:
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(lbl)
        else:
            # 중복이지만 qualifier가 다를 수 있으므로 병합
            existing = deduped[seen[key]]
            seen_q = {(q["qualifier"], q["qualifier_value"]) for q in existing.get("qualifiers", [])}
            for q in lbl.get("qualifiers", []):
                qk = (q["qualifier"], q["qualifier_value"])
                if qk not in seen_q:
                    existing.setdefault("qualifiers", []).append(q)
                    seen_q.add(qk)
            stats["removed"] += 1
    labels = deduped


    sample = dict(sample)
    sample["vertexSet"] = vertex_set
    sample["labels"] = labels
    return sample, stats

# --- LLM 호출 ---
def correct_document(doc_text: str, doc_title: str, existing_triples_str: str):
    global current_key_idx
    relation_menu_str = "\n".join([f"       - {pid}: {desc}" for pid, desc in RELATION_MAP.items()])

    prompt = f"""
    You are an Expert Knowledge Graph editor performing a CORRECTION pass.
    Your job is to fix errors in the existing triples extracted from the document below.
    Only return triples that need correction. Do NOT return triples that are already correct.

    [EXISTING TRIPLES]
    {existing_triples_str}

    [WHAT TO FIX]

    1. WRONG P-ID:
    - If a P-ID does not accurately describe the semantic relation between h and t, replace it with the correct one from the menu.
    - Example: P580 (start time) used for a person's career start → should be P571 or left as qualifier (WHEN).

    2. DUPLICATE VERTICES:
    - If two vertices refer to the same real-world entity, add to merge_vertices.
    - keep: the more complete/official name. remove: the shorter/less precise one.
    - Do NOT merge entities that are genuinely different (e.g. different time periods, rebranding).

    3. DUPLICATE EDGES:
    - If the same (h, r, t) triple appears more than once, add the duplicate to remove_labels.
    - Keep the one with more qualifiers.

    4. QUALIFIER CORRECTION:
    - Fix wrong 5W1H types: WHO (named agent/participant), WHAT (nature of relation), WHEN (time), WHERE (place), WHY (reason/cause), HOW (manner/role/capacity).
    - Remove qualifiers that restate the relation itself or are too vague (single word with no context).
    - qualifier_value MUST be within 5 words.
    - If the same edge has multiple qualifiers of the same 5W1H type, use the document context to decide:
      * If they represent the SAME information (e.g. two WHEREs for the same place, or a start+end time of the same period) → merge into one, keeping the more specific or combined value (e.g. "1892 to 1899").
      * If they represent GENUINELY DIFFERENT information (e.g. two unrelated WHY reasons, two different WHO participants) → keep both.

    5. DO NOT:
    - Add new triples (that is for a separate pass).
    - Change correct triples.
    - Return triples that need no correction.
    - If nothing needs correction, return empty lists for all fields.

    P-ID Relation Menu:
    {relation_menu_str}

    [Document Title]: {doc_title}
    [Document]
    {doc_text}
    """

    max_retries = len(API_KEYS)
    attempts = 0

    while attempts < max_retries:
        try:
            client = get_current_client()
            response = client.models.generate_content(
                model='models/gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=CorrectionResult,
                    temperature=0,
                ),
            )
            return response.parsed
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg or "403" in error_msg or "exhausted" in error_msg:
                print(f"\n  ⚠️ [키 {current_key_idx + 1} 한도 초과] 다음 키로 전환... ({e})")
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                attempts += 1
                time.sleep(2)
                continue
            elif "503" in error_msg or "unavailable" in error_msg or "overloaded" in error_msg:
                print(f"\n  🚧 [503] 5초 대기 후 재시도. ({e})")
                attempts += 1
                time.sleep(5)
                continue
            else:
                print(f"\n  ⚠️ [처리 불가능한 에러] 스킵: {e}")
                return None

    print("\n  ❌ 모든 API 키 소진.")
    return None

# --- 메인 ---
def process_correction_pass(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    start_idx = sum(1 for _ in open(output_file, 'r', encoding='utf-8')) if os.path.exists(output_file) else 0
    print(f"🚀 {start_idx}번 문서부터 수정 패스 시작 (총 {len(documents)}개)")

    total_merged = total_removed = total_r_changed = total_q_changed = 0

    for i in range(start_idx, len(documents)):
        sample = documents[i]
        doc_text = flatten_document(sample['sents'])
        existing_triples_str = format_existing_triples(sample['vertexSet'], sample['labels'])

        while True:
            print(f"[{i+1}/{len(documents)}] '{sample['title']}' 처리 중...", end=" ", flush=True)

            result = correct_document(doc_text, sample['title'], existing_triples_str)

            if result is not None:
                corrected_sample, stats = apply_corrections(sample, result)

                total_merged   += stats["merged"]
                total_removed  += stats["removed"]
                total_r_changed += stats["r_changed"]
                total_q_changed += stats["q_changed"]

                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(corrected_sample, ensure_ascii=False) + '\n')

                print(f"✅ (병합:{stats['merged']} 제거:{stats['removed']} P-ID교정:{stats['r_changed']} Q교정:{stats['q_changed']})")
                time.sleep(4.5)
                break
            else:
                print("❌ (API 실패 - 10초 후 재시도...)")
                time.sleep(10)

    print(f"\n✅ 수정 패스 완료: {len(documents)}개 문서")
    print(f"   버텍스 병합:  {total_merged}개")
    print(f"   엣지 제거:    {total_removed}개")
    print(f"   P-ID 교정:    {total_r_changed}개")
    print(f"   Qualifier 교정: {total_q_changed}개")

if __name__ == "__main__":
    INPUT_FILE  = "./Dataset/HypeRelDocRED/train/train_enriched_v4_clean.jsonl"
    OUTPUT_FILE = "./Dataset/HypeRelDocRED/train/train_enriched_v5.jsonl"

    process_correction_pass(INPUT_FILE, OUTPUT_FILE)