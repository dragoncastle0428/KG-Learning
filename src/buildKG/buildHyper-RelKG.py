import json
import os
import argparse
import webbrowser

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

COLOR_MAP = {
    "ORG":  {"bg": "#4a90d9", "font": "#ffffff"},
    "LOC":  {"bg": "#27ae60", "font": "#ffffff"},
    "MISC": {"bg": "#e67e22", "font": "#ffffff"},
    "PER":  {"bg": "#c0392b", "font": "#ffffff"},
    "TIME": {"bg": "#8e44ad", "font": "#ffffff"},
    "NUM":  {"bg": "#16a085", "font": "#ffffff"},
}

def load_doc(file_path: str, identifier) -> dict | None:
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            if isinstance(identifier, int):
                if i == identifier:
                    return doc
            else:
                if doc.get("title", "").strip().lower() == identifier.strip().lower():
                    return doc
    return None

def escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "")

def doc_to_vis_data(doc: dict, edge_color_default="#4a90d9", v4_edges=None):
    vertex_set = doc.get("vertexSet", [])
    labels = doc.get("labels", [])
    nodes, edges = [], []

    for i, entity_group in enumerate(vertex_set):
        if not entity_group:
            continue
        nd = entity_group[0]
        colors = COLOR_MAP.get(nd.get("type", "MISC"), {"bg": "#888888", "font": "#ffffff"})
        name = escape(nd["name"])
        typ  = nd.get("type", "MISC")
        nodes.append(
            "{"
            f"id:{i},"
            f"label:'{name}',"
            f"title:'Type: {typ}\\nName: {name}',"
            f"color:{{background:'{colors['bg']}',border:'#ffffff',highlight:{{background:'{colors['bg']}',border:'#ffff00'}}}},"
            f"font:{{color:'{colors['font']}',size:15,bold:true}},"
            f"shape:'ellipse',"
            f"margin:10"
            "}"
        )

    for ei, lbl in enumerate(labels):
        h_idx, t_idx = lbl["h"], lbl["t"]
        rel_text = escape(RELATION_MAP.get(lbl["r"], lbl["r"]))
        pid = lbl["r"]

        is_new = False
        if v4_edges is not None:
            h_name = vertex_set[h_idx][0]["name"].strip().lower()
            t_name = vertex_set[t_idx][0]["name"].strip().lower()
            is_new = (h_name, pid, t_name) not in v4_edges

        color = "#ff4444" if is_new else edge_color_default
        prefix = "🆕 " if is_new else ""
        tip_parts = [f"{prefix}Relation: {rel_text} ({pid})"]
        for q in lbl.get("qualifiers", []):
            tip_parts.append(f"[{q.get('qualifier','?')}]: {escape(q.get('qualifier_value',''))}")
        tip = "\\n".join(tip_parts)

        edges.append(
            "{"
            f"id:{ei},"
            f"from:{h_idx},"
            f"to:{t_idx},"
            f"label:'{rel_text}',"
            f"title:'{tip}',"
            f"color:{{color:'{color}',highlight:'{color}',hover:'{color}'}},"
            f"font:{{color:'white',size:12,strokeWidth:4,strokeColor:'#000000'}},"
            f"arrows:'to'"
            "}"
        )

    return "[" + ",\n".join(nodes) + "]", "[" + ",\n".join(edges) + "]"

def make_script(div_id, nodes_js, edges_js):
    return f"""(function(){{
  var c = document.getElementById('{div_id}');
  var n = new vis.DataSet({nodes_js});
  var e = new vis.DataSet({edges_js});
  new vis.Network(c, {{nodes:n, edges:e}}, {{
    physics:{{
      forceAtlas2Based:{{gravitationalConstant:-80,centralGravity:0.01,springLength:120}},
      solver:'forceAtlas2Based',
      stabilization:{{iterations:200}}
    }},
    edges:{{smooth:{{type:'dynamic'}}}},
    interaction:{{hover:true,tooltipDelay:50}}
  }});
}})();"""

VIS_CDN = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css">
"""

def build_html_single(doc):
    title = doc.get("title", "KG")
    nodes_js, edges_js = doc_to_vis_data(doc)
    script = make_script("graph_main", nodes_js, edges_js)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
{VIS_CDN}
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#1a1a1a}}
#graph_main{{width:100vw;height:100vh}}</style></head>
<body><div id="graph_main"></div><script>{script}</script></body></html>"""

def build_html_compare(doc_v4, doc_v5, v4_edges, new_count):
    title = doc_v4.get("title", "Unknown")
    n4, e4 = doc_to_vis_data(doc_v4, "#4a90d9")
    n5, e5 = doc_to_vis_data(doc_v5, "#4a90d9", v4_edges=v4_edges)
    s4 = make_script("graph_v4", n4, e4)
    s5 = make_script("graph_v5", n5, e5)
    return f"""<!DOCTYPE html><html>
<head><meta charset="utf-8"><title>KG 비교: {title}</title>
{VIS_CDN}
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#111;color:white;font-family:sans-serif;display:flex;flex-direction:column;height:100vh;overflow:hidden}}
.header{{padding:10px;text-align:center;flex-shrink:0;background:#0d0d0d}}
.header h1{{font-size:17px;color:#eeeeee;margin-bottom:4px}}
.legend{{font-size:13px}}.legend span{{margin:0 14px}}
.panels{{display:flex;flex:1;min-height:0}}
.panel{{flex:1;display:flex;flex-direction:column;border:1px solid #333;min-width:0}}
.panel-title{{text-align:center;font-size:13px;padding:6px;background:#1a1a1a;flex-shrink:0;font-weight:bold}}
#graph_v4,#graph_v5{{flex:1;min-height:0}}
</style></head>
<body>
<div class="header">
  <h1>📊 KG 비교: {title}</h1>
  <div class="legend">
    <span style="color:#4a90d9">● 기존 엣지 (v4)</span>
    <span style="color:#ff4444">● 새로 추가/수정된 엣지 (v5)</span>
  </div>
</div>
<div class="panels">
  <div class="panel">
    <div class="panel-title" style="color:#7ab8f5">v4 (원본) — {len(doc_v4['labels'])}개 엣지 / {len(doc_v4['vertexSet'])}개 버텍스</div>
    <div id="graph_v4"></div>
  </div>
  <div class="panel">
    <div class="panel-title" style="color:#7ddb8a">v5 (수정) — {len(doc_v5['labels'])}개 엣지 / {len(doc_v5['vertexSet'])}개 버텍스 &nbsp;|&nbsp; 변경: {new_count}개</div>
    <div id="graph_v5"></div>
  </div>
</div>
<script>{s4}{s5}</script>
</body></html>"""

def visualize_single(file_path, identifier, output_html):
    doc = load_doc(file_path, identifier)
    if not doc:
        print(f"❌ 문서를 찾을 수 없습니다: {identifier}"); return
    print(f"[{doc.get('title')}] 시각화 중...")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(build_html_single(doc))
    abs_path = os.path.abspath(output_html)
    print(f"✅ {abs_path}")
    webbrowser.open(f"file:///{abs_path}")

def visualize_compare(file_v4, file_v5, identifier, output_html):
    doc_v4 = load_doc(file_v4, identifier)
    doc_v5 = load_doc(file_v5, identifier)
    if not doc_v4:
        print(f"❌ v4 문서 없음: {identifier}"); return
    if not doc_v5:
        print(f"❌ v5 문서 없음: {identifier}"); return
    title = doc_v4.get("title", "Unknown")
    print(f"[{title}] 비교 중...")
    v4_vset = doc_v4["vertexSet"]
    v4_edges = {
        (v4_vset[l["h"]][0]["name"].strip().lower(), l["r"],
         v4_vset[l["t"]][0]["name"].strip().lower())
        for l in doc_v4["labels"]
    }
    v5_vset = doc_v5["vertexSet"]
    new_count = sum(
        1 for l in doc_v5["labels"]
        if (v5_vset[l["h"]][0]["name"].strip().lower(), l["r"],
            v5_vset[l["t"]][0]["name"].strip().lower()) not in v4_edges
    )
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(build_html_compare(doc_v4, doc_v5, v4_edges, new_count))
    abs_path = os.path.abspath(output_html)
    print(f"✅ {abs_path}")
    print(f"   v4: {len(doc_v4['labels'])}개 엣지 | v5: {len(doc_v5['labels'])}개 엣지 | 변경: {new_count}개")
    webbrowser.open(f"file:///{abs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "compare"], default="compare")
    parser.add_argument("--file", default="./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl")
    parser.add_argument("--v4",   default="./Dataset/HypeRelDocRED/train/train_enriched_v4.jsonl")
    parser.add_argument("--v5",   default="./Dataset/HypeRelDocRED/train/train_enriched_v5.jsonl")
    parser.add_argument("--doc",  default="18", help="줄 번호(숫자) 또는 title 문자열")
    parser.add_argument("--output", default="kg_compare.html")
    args = parser.parse_args()
    try:
        doc_id = int(args.doc)
    except ValueError:
        doc_id = args.doc
    if args.mode == "single":
        visualize_single(args.file, doc_id, args.output)
    else:
        visualize_compare(args.v4, args.v5, doc_id, args.output)