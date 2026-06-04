import os
import json
import time
import re
import difflib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

VALID_QUALIFIER_TYPES = {"WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"}

load_dotenv()
API_KEYS = [os.getenv(f"GEMINI_API_KEY{i}") for i in range(1, 41) if os.getenv(f"GEMINI_API_KEY{i}")]
current_key_idx = 0

def get_current_client():
    global current_key_idx
    return Client(api_key=API_KEYS[current_key_idx])

# ── 스키마: LLM Call 1 (누락 추가 + MISC 처리) ───────────────────────────────

QualifierType = Literal["WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"]
EntityType    = Literal["ORG", "LOC", "PER", "TIME", "NUM", "EVENT", "MISC"]

class Qualifier(BaseModel):
    qualifier: QualifierType
    qualifier_value: str = Field(description="Contextual detail (within 7 words)")

class NewLabel(BaseModel):
    h: str = Field(description="Exact head entity text from the document")
    h_type: EntityType
    r: str  = Field(description="Wikidata P-ID from the relation menu")
    t: str  = Field(description="Exact tail entity text from the document")
    t_type: EntityType
    qualifiers: List[Qualifier] = Field(default_factory=list)

class MiscDemotion(BaseModel):
    misc_entity: str  = Field(description="MISC entity name to demote")
    attach_to_h: str  = Field(description="Head entity of the triple to attach qualifier to")
    attach_to_r: str  = Field(description="P-ID of the triple to attach qualifier to")
    attach_to_t: str  = Field(description="Tail entity of the triple to attach qualifier to")
    qualifier: Qualifier

class MiscRetype(BaseModel):
    misc_entity: str = Field(description="MISC entity name to retype")
    new_type: Literal["EVENT", "ORG", "LOC", "PER", "TIME", "NUM"]
    reason: str      = Field(description="One sentence reason")

class Call1Result(BaseModel):
    new_labels:     List[NewLabel]     = Field(default_factory=list)
    misc_demotions: List[MiscDemotion] = Field(default_factory=list)
    misc_retypes:   List[MiscRetype]   = Field(default_factory=list)

# ── 스키마: LLM Call 2 (교정 패스) ───────────────────────────────────────────

class CorrectedLabel(BaseModel):
    h: str     = Field(description="Exact head entity name from existing triples (for matching)")
    r: str     = Field(description="Original P-ID (for matching)")
    t: str     = Field(description="Exact tail entity name from existing triples (for matching)")
    new_r: str = Field(description="Corrected P-ID. Same as r if no change needed.")
    qualifiers: List[Qualifier] = Field(description="Corrected full qualifier list for this edge")

class MergeVertex(BaseModel):
    keep:   str = Field(description="Entity name to keep")
    remove: str = Field(description="Entity name to remove and merge into keep")

class RemoveLabel(BaseModel):
    h: str = Field(description="Head entity name of the edge to remove")
    r: str = Field(description="P-ID of the edge to remove")
    t: str = Field(description="Tail entity name of the edge to remove")

class Call2Result(BaseModel):
    corrected_labels: List[CorrectedLabel] = Field(default_factory=list)
    merge_vertices:   List[MergeVertex]    = Field(default_factory=list)
    remove_labels:    List[RemoveLabel]    = Field(default_factory=list)

# ── 유틸 ─────────────────────────────────────────────────────────────────────

def flatten_document(sents: list) -> str:
    return " ".join([" ".join(sent) for sent in sents])

def format_triples(vertex_set: list, labels: list) -> str:
    lines = []
    for lbl in labels:
        h_name = vertex_set[lbl["h"]][0]["name"]
        h_type = vertex_set[lbl["h"]][0]["type"]
        t_name = vertex_set[lbl["t"]][0]["name"]
        t_type = vertex_set[lbl["t"]][0]["type"]
        r_desc = RELATION_MAP.get(lbl["r"], lbl["r"])
        qs     = lbl.get("qualifiers", [])
        q_str  = ", ".join(f"{q['qualifier']}: {q['qualifier_value']}" for q in qs) or "none"
        lines.append(
            f"  ({h_name} [{h_type}]) --[{lbl['r']}: {r_desc}]--> ({t_name} [{t_type}]) | qualifiers: [{q_str}]"
        )
    return "\n".join(lines)

def get_misc_entities(vertex_set: list, labels: list) -> list[dict]:
    misc_list, seen = [], set()
    for lbl in labels:
        for role, idx in [("h", lbl["h"]), ("t", lbl["t"])]:
            entity = vertex_set[idx][0]
            if entity["type"] == "MISC" and entity["name"] not in seen:
                seen.add(entity["name"])
                h_name = vertex_set[lbl["h"]][0]["name"]
                t_name = vertex_set[lbl["t"]][0]["name"]
                misc_list.append({
                    "name":   entity["name"],
                    "role":   role,
                    "triple": f"({h_name}, {lbl['r']}, {t_name})"
                })
    return misc_list

def clean_text(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

def find_pos(sents: list, target: str) -> dict:
    target_tokens = target.lower().split()
    target_len    = len(target_tokens)
    for si, sent in enumerate(sents):
        lower = [t.lower() for t in sent]
        for i in range(len(lower) - target_len + 1):
            if lower[i:i+target_len] == target_tokens:
                return {"sent_id": si, "pos": [i, i + target_len]}
    target_clean = clean_text(target)
    best = {"sent_id": -1, "pos": [-1, -1], "score": 0.0}
    for si, sent in enumerate(sents):
        for ws in [target_len - 1, target_len, target_len + 1]:
            if ws <= 0:
                continue
            for i in range(len(sent) - ws + 1):
                w   = " ".join(clean_text(t) for t in sent[i:i+ws]).strip()
                scr = difflib.SequenceMatcher(None, target_clean, w).ratio()
                if scr > best["score"]:
                    best = {"sent_id": si, "pos": [i, i + ws], "score": scr}
    return best if best["score"] >= 0.80 else {"sent_id": -1, "pos": [-1, -1]}

def normalize_qualifiers(qualifiers: list) -> list:
    """qualifier 타입 대소문자 정규화 + 유효하지 않은 타입 필터링"""
    result = []
    for q in qualifiers:
        q_type = q.get("qualifier", "").upper()
        if q_type in VALID_QUALIFIER_TYPES:
            result.append({"qualifier": q_type, "qualifier_value": q["qualifier_value"]})
    return result

# ── 후처리: Call 1 결과 적용 ─────────────────────────────────────────────────

def apply_call1(sample: dict, result: Call1Result) -> tuple[dict, dict]:
    vertex_set = [list(g) for g in sample["vertexSet"]]
    labels     = [dict(l) for l in sample["labels"]]
    stats      = {"added": 0, "demoted": 0, "retyped": 0}

    name_to_idx = {vg[0]["name"].strip().lower(): i for i, vg in enumerate(vertex_set)}

    def get_or_create(name: str, etype: str) -> int:
        key = name.strip().lower()
        if key in name_to_idx:
            return name_to_idx[key]
        loc = find_pos(sample["sents"], name)
        vertex_set.append([{"name": name.strip(), "type": etype,
                             "sent_id": loc["sent_id"], "pos": loc["pos"]}])
        idx = len(vertex_set) - 1
        name_to_idx[key] = idx
        return idx

    # 1. 누락 관계 추가
    existing_keys = {(l["h"], l["r"], l["t"]) for l in labels}
    for nl in result.new_labels:
        if nl.r not in RELATION_MAP:
            continue
        h_idx = get_or_create(nl.h, nl.h_type)
        t_idx = get_or_create(nl.t, nl.t_type)
        key   = (h_idx, nl.r, t_idx)
        if key in existing_keys or h_idx == t_idx:
            continue
        h_sent = vertex_set[h_idx][0]["sent_id"]
        t_sent = vertex_set[t_idx][0]["sent_id"]
        ev     = sorted({s for s in [h_sent, t_sent] if s != -1})
        labels.append({
            "h": h_idx, "r": nl.r, "t": t_idx,
            "evidence": ev,
            "qualifiers": normalize_qualifiers([q.model_dump() for q in nl.qualifiers])
        })
        existing_keys.add(key)
        stats["added"] += 1

    # 2. MISC → qualifier 강등
    for dm in result.misc_demotions:
        h_key = dm.attach_to_h.strip().lower()
        t_key = dm.attach_to_t.strip().lower()
        if h_key not in name_to_idx or t_key not in name_to_idx:
            continue
        h_idx = name_to_idx[h_key]
        t_idx = name_to_idx[t_key]
        for lbl in labels:
            if lbl["h"] == h_idx and lbl["r"] == dm.attach_to_r and lbl["t"] == t_idx:
                new_q  = dm.qualifier.model_dump()
                new_q["qualifier"] = new_q["qualifier"].upper()
                if new_q["qualifier"] not in VALID_QUALIFIER_TYPES:
                    break
                seen_q = {(q["qualifier"], q["qualifier_value"])
                          for q in lbl.get("qualifiers", [])}
                if (new_q["qualifier"], new_q["qualifier_value"]) not in seen_q:
                    lbl.setdefault("qualifiers", []).append(new_q)
                misc_key = dm.misc_entity.strip().lower()
                if misc_key in name_to_idx:
                    misc_idx = name_to_idx[misc_key]
                    labels[:] = [
                        l for l in labels
                        if not (l["h"] == misc_idx or l["t"] == misc_idx)
                        or (l["h"] == h_idx and l["r"] == dm.attach_to_r and l["t"] == t_idx)
                    ]
                stats["demoted"] += 1
                break

    # 3. MISC → 타입 재분류
    for rt in result.misc_retypes:
        key = rt.misc_entity.strip().lower()
        if key in name_to_idx:
            vertex_set[name_to_idx[key]][0]["type"] = rt.new_type
            stats["retyped"] += 1

    sample = dict(sample)
    sample["vertexSet"] = vertex_set
    sample["labels"]    = labels
    return sample, stats

# ── 후처리: Call 2 결과 적용 ─────────────────────────────────────────────────

def apply_call2(sample: dict, result: Call2Result) -> tuple[dict, dict]:
    vertex_set = [list(g) for g in sample["vertexSet"]]
    labels     = [dict(l) for l in sample["labels"]]
    stats      = {"merged": 0, "removed": 0, "r_changed": 0, "q_changed": 0}

    name_to_idx = {vg[0]["name"].strip().lower(): i for i, vg in enumerate(vertex_set)}

    # 1. 버텍스 병합
    remap = {}
    for mv in result.merge_vertices:
        keep_key   = mv.keep.strip().lower()
        remove_key = mv.remove.strip().lower()
        if keep_key in name_to_idx and remove_key in name_to_idx:
            remap[name_to_idx[remove_key]] = name_to_idx[keep_key]
            stats["merged"] += 1

    def resolve(idx):
        while idx in remap:
            idx = remap[idx]
        return idx

    for lbl in labels:
        lbl["h"] = resolve(lbl["h"])
        lbl["t"] = resolve(lbl["t"])

    # 2. 중복 엣지 제거 (remove_labels) — 버그픽스: 유지 엣지 없으면 복원
    remove_set = set()
    for rl in result.remove_labels:
        h_key = rl.h.strip().lower()
        t_key = rl.t.strip().lower()
        if h_key in name_to_idx and t_key in name_to_idx:
            h_idx = resolve(name_to_idx[h_key])
            t_idx = resolve(name_to_idx[t_key])
            remove_set.add((h_idx, rl.r, t_idx))

    keep_map  = {}
    merge_q   = {}
    lbl_by_key = {}
    for lbl in labels:
        key = (lbl["h"], lbl["r"], lbl["t"])
        lbl_by_key[key] = lbl
        if key in remove_set:
            merge_q.setdefault(key, []).extend(lbl.get("qualifiers", []))
        else:
            keep_map[key] = lbl

    for key, extra_quals in merge_q.items():
        if key in keep_map:
            existing = keep_map[key]
            seen_q   = {(q["qualifier"], q["qualifier_value"])
                        for q in existing.get("qualifiers", [])}
            for q in extra_quals:
                qk = (q["qualifier"], q["qualifier_value"])
                if qk not in seen_q:
                    existing.setdefault("qualifiers", []).append(q)
                    seen_q.add(qk)
            stats["removed"] += 1
        else:
            # 유지할 엣지 없으면 삭제 말고 복원
            keep_map[key] = lbl_by_key[key]

    labels = list(keep_map.values())

    # 3. 엣지 교정 + qualifier 정규화
    correction_map = {}
    for cl in result.corrected_labels:
        h_key = cl.h.strip().lower()
        t_key = cl.t.strip().lower()
        if h_key in name_to_idx and t_key in name_to_idx:
            h_idx = resolve(name_to_idx[h_key])
            t_idx = resolve(name_to_idx[t_key])
            correction_map[(h_idx, cl.r, t_idx)] = cl

    for lbl in labels:
        # qualifier 정규화 (전체 적용)
        lbl["qualifiers"] = normalize_qualifiers(lbl.get("qualifiers", []))
        # 교정 적용
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key in correction_map:
            cl = correction_map[key]
            if cl.new_r != lbl["r"] and cl.new_r in RELATION_MAP:
                lbl["r"] = cl.new_r
                stats["r_changed"] += 1
            old_q = lbl.get("qualifiers", [])
            new_q = normalize_qualifiers([q.model_dump() for q in cl.qualifiers])
            if old_q != new_q:
                lbl["qualifiers"] = new_q
                stats["q_changed"] += 1

    # 4. 셀프 루프 제거
    labels = [l for l in labels if l["h"] != l["t"]]

    # 5. 최종 중복 병합 (버텍스 병합 후 생긴 중복)
    seen, deduped = {}, []
    for lbl in labels:
        key = (lbl["h"], lbl["r"], lbl["t"])
        if key not in seen:
            seen[key] = len(deduped)
            deduped.append(lbl)
        else:
            existing = deduped[seen[key]]
            seen_q   = {(q["qualifier"], q["qualifier_value"])
                        for q in existing.get("qualifiers", [])}
            for q in lbl.get("qualifiers", []):
                qk = (q["qualifier"], q["qualifier_value"])
                if qk not in seen_q:
                    existing.setdefault("qualifiers", []).append(q)
                    seen_q.add(qk)
            stats["removed"] += 1

    sample = dict(sample)
    sample["vertexSet"] = vertex_set
    sample["labels"]    = deduped
    return sample, stats

# ── LLM 호출 공통 래퍼 ────────────────────────────────────────────────────────

def llm_call(prompt: str, schema, timeout: int = 120) -> Optional[object]:
    global current_key_idx
    max_retries = len(API_KEYS)
    attempts    = 0

    while attempts < max_retries:
        try:
            client = get_current_client()

            def _call():
                return client.models.generate_content(
                    model="models/gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                try:
                    response = future.result(timeout=timeout)
                except FuturesTimeoutError:
                    print(f"\n  ⏱️ [타임아웃 {timeout}s] 다음 키로 전환...")
                    current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                    attempts += 1
                    continue

            return response.parsed

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["429", "quota", "403", "exhausted"]):
                print(f"\n  ⚠️ [키 {current_key_idx+1} 한도 초과] 다음 키로 전환... ({e})")
                current_key_idx = (current_key_idx + 1) % len(API_KEYS)
                attempts += 1
                time.sleep(2)
            elif any(x in err for x in ["503", "unavailable", "overloaded"]):
                print(f"\n  🚧 [503] 5초 대기 후 재시도. ({e})")
                attempts += 1
                time.sleep(5)
            else:
                print(f"\n  ⚠️ [처리 불가능한 에러] 스킵: {e}")
                return None

    print("\n  ❌ 모든 API 키 소진.")
    return None

# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────

def build_call1_prompt(doc_text: str, doc_title: str,
                       existing_triples_str: str, misc_entities: list[dict]) -> str:
    relation_menu = "\n".join(f"  - {pid}: {desc}" for pid, desc in RELATION_MAP.items())
    misc_str = "\n".join(
        f"  - \"{m['name']}\" (appears as {m['role']} in {m['triple']})"
        for m in misc_entities
    ) or "  (none)"

    return f"""
You are a Knowledge Graph auditor performing two tasks on the document below.

════════════════════════════════════════════
TASK A — ADD MISSING RELATIONS
════════════════════════════════════════════
[EXISTING TRIPLES]
{existing_triples_str}

Rules for new_labels:
- Only add if the relation is explicitly stated in the text.
- Do NOT re-extract or rephrase existing triples.
- Do NOT add generic located-in / instance-of chains unless they carry
  meaningful new information beyond what is already captured.
- Every new triple MUST have at least one qualifier.
- h and t MUST be exact entity names from the text (≤5 words unless proper noun).
- r MUST be a P-ID from the relation menu below.
- Entity types: ORG / LOC / PER / TIME / NUM / EVENT / MISC

════════════════════════════════════════════
TASK B — HANDLE MISC ENTITIES
════════════════════════════════════════════
[MISC ENTITIES]
{misc_str}

For each MISC entity, choose EXACTLY ONE action:

① DEMOTE TO QUALIFIER (misc_demotions)
   → Only if the entity solely describes one specific relation and does not
     appear independently elsewhere in the document.
   → 5W1H types: WHO / WHAT / WHEN / WHERE / WHY / HOW
   → Specify which triple absorbs it (attach_to_h, attach_to_r, attach_to_t).

② RETYPE (misc_retypes)
   → If clearly belongs to: EVENT / ORG / LOC / PER / TIME / NUM
   → Provide a one-sentence reason.

③ KEEP AS MISC — simply omit from both lists.

════════════════════════════════════════════
RELATION MENU
════════════════════════════════════════════
{relation_menu}

[Document Title]: {doc_title}
[Document]
{doc_text}
"""

def build_call2_prompt(doc_text: str, doc_title: str,
                       existing_triples_str: str) -> str:
    relation_menu = "\n".join(f"  - {pid}: {desc}" for pid, desc in RELATION_MAP.items())

    return f"""
You are a Knowledge Graph editor performing a CORRECTION pass.
Fix errors in the existing triples. Only return triples that need correction.

[EXISTING TRIPLES]
{existing_triples_str}

[WHAT TO FIX]

1. WRONG P-ID:
   - Replace with the correct one from the menu if semantically wrong.

2. DUPLICATE VERTICES:
   - If two vertices refer to the same real-world entity → merge_vertices.
   - keep: more complete/official name. remove: shorter/less precise.
   - Do NOT merge genuinely different entities (e.g. different time periods).

3. DUPLICATE EDGES:
   - Same (h, r, t) appearing more than once → add duplicate to remove_labels.
   - Keep the one with more qualifiers.

4. QUALIFIER CORRECTION:
   - Fix wrong 5W1H types (WHO/WHAT/WHEN/WHERE/WHY/HOW — uppercase only).
   - Remove qualifiers that restate the relation or are too vague.
   - qualifier_value MUST be within 7 words.
   - Same-type qualifiers: merge if same info, keep both if genuinely different.

5. DO NOT add new triples or change correct triples.
   If nothing needs correction, return empty lists.

[RELATION MENU]
{relation_menu}

[Document Title]: {doc_title}
[Document]
{doc_text}
"""

# ── 메인 ─────────────────────────────────────────────────────────────────────

def process(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    start_idx = (sum(1 for _ in open(output_file, encoding="utf-8"))
                 if os.path.exists(output_file) else 0)
    print(f"🚀 {start_idx}번 문서부터 시작 (총 {len(documents)}개)")

    totals = {"added": 0, "demoted": 0, "retyped": 0,
              "merged": 0, "removed": 0, "r_changed": 0, "q_changed": 0}

    for i in range(start_idx, len(documents)):
        sample = documents[i]
        doc_text = flatten_document(sample["sents"])

        while True:
            print(f"[{i+1}/{len(documents)}] '{sample['title']}'", end=" ", flush=True)

            # ── LLM Call 1: 누락 추가 + MISC 처리 ──
            c1_prompt = build_call1_prompt(
                doc_text, sample["title"],
                format_triples(sample["vertexSet"], sample["labels"]),
                get_misc_entities(sample["vertexSet"], sample["labels"])
            )
            result1 = llm_call(c1_prompt, Call1Result)
            if result1 is None:
                print("❌ Call1 실패 - 10초 후 재시도...")
                time.sleep(10)
                continue

            sample_after_c1, stats1 = apply_call1(sample, result1)
            print(f"[C1 추가:{stats1['added']} 강등:{stats1['demoted']} 재분류:{stats1['retyped']}]", end=" ", flush=True)

            # ── LLM Call 2: 교정 패스 ──
            c2_prompt = build_call2_prompt(
                doc_text, sample["title"],
                format_triples(sample_after_c1["vertexSet"], sample_after_c1["labels"])
            )
            result2 = llm_call(c2_prompt, Call2Result)
            if result2 is None:
                print("❌ Call2 실패 - 10초 후 재시도...")
                time.sleep(10)
                continue

            final_sample, stats2 = apply_call2(sample_after_c1, result2)
            print(f"[C2 병합:{stats2['merged']} 제거:{stats2['removed']} P-ID:{stats2['r_changed']} Q:{stats2['q_changed']}] ✅")

            for k in stats1:
                totals[k] += stats1[k]
            for k in stats2:
                totals[k] += stats2[k]

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_sample, ensure_ascii=False) + "\n")

            time.sleep(4.5)
            break

    print(f"\n✅ 완료: {len(documents)}개 문서")
    print(f"   [Call 1] 관계 추가: {totals['added']} | MISC 강등: {totals['demoted']} | 재분류: {totals['retyped']}")
    print(f"   [Call 2] 병합: {totals['merged']} | 제거: {totals['removed']} | P-ID 교정: {totals['r_changed']} | Q 교정: {totals['q_changed']}")


if __name__ == "__main__":
    INPUT_FILE  = "./Dataset/HypeRelDocRED/train/train_enriched_v4_clean.jsonl"
    OUTPUT_FILE = "./Dataset/HypeRelDocRED/train/train_enriched_v6.jsonl"
    process(INPUT_FILE, OUTPUT_FILE)