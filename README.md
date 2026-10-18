# HypeRelDocRED: Does Hyper-Relational Context Help GraphRAG?

> **Status: paused (June 2026), unfinished.** This was a personal research project carried out in the first half of 2026. The dataset is not complete and the evaluation is small-scale and LLM-judged. Treat the numbers below as preliminary evidence, not a benchmark result.

## Motivation

Document-level relation extraction datasets such as [Re-DocRED](https://github.com/tonytan48/Re-DocRED) store facts as plain triples `(head, relation, tail)`. Much of the context in the source text — *when* something happened, *where*, *why*, *how* — is lost.

The hypothesis tested here:

> If each triple carries its contextual **qualifiers**, a GraphRAG system built on that graph should answer questions more accurately than one built on the plain-triple graph — especially questions that depend on that context.

To test it, I used an LLM API to enrich Re-DocRED into a hyper-relational version (**HypeRelDocRED**), loaded both versions into Neo4j, and compared GraphRAG question answering on the two graphs.

## Schema

Every fact is a base triple plus a list of qualifiers:

```
(head, relation, tail)  +  [(qualifier_type, qualifier_value), ...]
```

- **Relations** are Wikidata property IDs (P-IDs) from a fixed menu of ~100 properties.
- **Qualifier types** follow a 5W1H schema: `WHO`, `WHAT`, `WHEN`, `WHERE`, `WHY`, `HOW`.
- **Qualifier values** are short spans (≤ 7 words) taken from the document.
- **Entity types**: `PER`, `ORG`, `LOC`, `TIME`, `NUM`, `EVENT`, `MISC`.

All LLM calls use **Gemini 2.5 Flash** with structured output enforced by Pydantic schemas.

## How the dataset evolved

The enrichment was built iteratively. Each round exposed problems that shaped the next one.

### v2 – v4: single-pass enrichment

Each Re-DocRED document and its existing triples were sent to the LLM, which attached qualifiers and added triples the original annotation had missed (`Dataset/creatingNewData.py`, `creatingNewData2.py`). Early versions used a looser qualifier set (including `ROLE`); v4 settled on 5W1H.

Inspecting the v4 graphs showed recurring problems:

- **Schema drift** — qualifiers outside 5W1H, and values with no information (`formerly`, `now`, `today`)
- **Invalid or wrong P-IDs** — relations mapped to non-existent or semantically wrong properties
- **Duplicates** — the same `(h, r, t)` edge more than once, and the same entity under several names (`Commonwealth` / `Commonwealth of Nations`)
- **Dropped edges** — forcing every fact onto a P-ID made the model skip relations it couldn't map
- **Fragmented graphs** — disconnected components, so a document no longer read as one graph

### v5: correction pass

A second LLM call takes the existing triples and returns **only the ones that need fixing**: wrong P-IDs, duplicate vertices to merge, duplicate edges to remove, and wrong or vague qualifiers. It is explicitly told not to add new triples, to keep each call focused. A rule-based step (`Dataset/postProcess.py`) then normalises qualifier types, drops invalid ones and removes isolated nodes.

![v4 vs v5](docs/v4_vs_v5.png)
*Left: v4 (14 edges, 14 vertices). Right: v5 after the correction pass (13 edges, 12 vertices). Duplicate entities are merged and the separate "Ambassador" subgraph is reconnected to New Zealand.*

### v6: second pass for missing relations

GraphRAG testing showed that many answers failed because a relation was never extracted — the LLM had judged it "unimportant". v6 (`Dataset/creatingNewData3.py`, built from `v4_clean`) runs two calls per document:

1. **Recall pass** — add missing relations. If a missing fact fits as a qualifier on an existing edge, attach it there; otherwise create a new triple. Vague `MISC` entities are either retyped (e.g. to `EVENT`) or demoted to a qualifier.
2. **Correction pass** — fix P-IDs, merge duplicate edges (taking into account edges that differ only in a qualifier, such as two `WHEN` values), unify entities, and remove unsupported edges.

![v4 vs v6](docs/v4_vs_v6.png)
*Left: v4. Right: v6. The isolated node is connected, and dates, genre and award nominations that v4 missed are now in the graph.*

### Dataset versions

Only the `train` split has been processed (Re-DocRED train has ~3,000 documents).

| File (`Dataset/HypeRelDocRED/train/`) | Documents | Triples | Qualifiers | Notes |
|---|---:|---:|---:|---|
| `train_enriched_v2.jsonl` | 743 | 8,965 | 6,086 | early qualifier set |
| `train_enriched_v3.jsonl` | 584 | 6,362 | 4,383 | early qualifier set |
| `train_enriched_v4.jsonl` | 2,970 | 41,488 | 42,270 | 5W1H; used for the LLM-judge evaluations below |
| `train_enriched_v4_clean.jsonl` | 2,970 | 41,465 | 38,793 | v4 after cleaning |
| `train_enriched_v5.jsonl` | 1,080 | 14,386 | 12,827 | correction pass (partial) |
| `train_enriched_v5_clean.jsonl` | 1,053 | 14,017 | 12,483 | v5 after `postProcess.py` |
| `train_enriched_v6.jsonl` | 1,512 | 23,399 | 22,997 | recall + correction passes; **partial** (stopped when the project paused) |

## Evaluation

### Knowledge graphs

Both datasets were loaded into Neo4j as separate databases — `original` (plain Re-DocRED triples) and `hyper` (HypeRelDocRED, qualifiers stored on edges) — using `src/KGConstruction/`. `src/buildKG/` renders single documents as interactive HTML graphs (vis.js); examples are in `KGs/`, `results/` and `kg_compare.html`.

### GraphRAG QA pipeline (`src/evalDataset/graphRAG.py`)

1. Sample documents and have the LLM write three question types: **Simple**, **Multi-hop**, and **Qualifier** (needs time / place / reason context).
2. Extract entities from each question and retrieve a subgraph from each database with Cypher.
3. Answer each question from each subgraph.
4. An LLM judge scores both answers (0–10) and picks a winner.

### Results

**Dataset quality** (`src/evalDataset/evaluate_dataset.py`) — the LLM judge scored the Re-DocRED, v2 and v4 graphs of the same 45 documents against the source text (0–100, higher is better).

| Criterion | Re-DocRED | v2 | v4 |
|---|---:|---:|---:|
| Hallucination (higher = fewer) | 50.7 | 88.2 | 95.6 |
| Comprehensiveness | 35.4 | 73.7 | 93.5 |
| Schema mapping | 50.2 | 79.8 | 95.4 |
| Salience | 45.1 | 84.7 | 97.2 |
| Entity clarity | 67.7 | 91.1 | 96.2 |

**GraphRAG QA** — 12 runs × 9 questions = 108 questions (0–10).

| Question type | n | Original avg | Hyper avg | Hyper wins | Original wins | Ties |
|---|---:|---:|---:|---:|---:|---:|
| Simple | 36 | 2.99 | 6.43 | 20 | 6 | 10 |
| Multi-hop | 36 | 1.12 | 3.78 | 14 | 2 | 20 |
| Qualifier | 36 | 0.75 | 3.81 | 13 | 2 | 21 |
| **Total** | **108** | **1.62** | **4.67** | **47** | **10** | **51** |

The 12 runs span several iterations of the retrieval code; on the final pipeline alone (4 runs, 36 questions) the result was 13 Hyper wins, 3 Original wins and 20 ties.

**v6 check** — on a 27-question test after the v6 second pass:

- Setting aside retrieval failures, no question failed because the needed relation was missing from the graph.
- For Qualifier and Multi-hop questions, the share of questions where the answer was found rose from roughly **20% to 70%**.

Example questions from this test: *"What was the worldwide sales figure for Maeve Binchy's novels, which were translated into 37 languages?"* (more than 40 million copies) and *"In which country is the last team Henrik Malmström played for located?"* (Norway). In the judged runs, the original graph often returned nothing relevant, while the hyper-relational graph supplied the detail the question hinged on — for example, that a player's time at a club was a brief spell in 2007.

## Lessons learned

- **Extract relations in two passes.** One pass for the important relations and one for all relations, then merge, works better than asking for everything at once.
- **Do entity unification as post-processing.** It needs the whole document's graph in view; string or embedding similarity alone is too weak. Explicit rules would help (e.g. keep the longest official name for organisations).
- **Split the work into small, focused LLM calls.** Long prompts that asked for too much at once led to ignored instructions and hallucinations — made worse by working within free-tier API limits.
- **A strict schema matters.** Fixed P-IDs, 5W1H qualifier types and length-limited values made the output much easier to validate and clean.
- **Retrieval is now the bottleneck.** Most remaining failures come from Cypher retrieval missing data that exists, often because an entity's name was changed during unification (abbreviations, dropped middle names).

## Limitations

- **The dataset is incomplete.** Only part of the train split has the latest pipeline applied, and the enrichment is LLM-generated with no systematic human validation.
- **LLM-as-judge bias.** The same model family generated the enrichment, the questions and the judgements. In the dataset-quality evaluation the judge also saw which graph was the newest version.
- **Small sample.** 45 documents and 108 + 27 questions, with questions generated from the hyper-relational graph itself.
- **Low absolute scores and many ties.** Both systems often failed, which points at retrieval as much as at graph content.
- **No standard baselines** such as text-chunk RAG or existing GraphRAG frameworks.

## If the project is resumed

- Finish v6 over the full train split, then the validation and test splits
- Improve Cypher retrieval (entity matching that tolerates name variants)
- Clearer entity unification rules
- Try a skill-driven extraction approach (an LLM that distils reusable extraction rules from its own errors) on hyper-relational KGs
- Evaluate on an independent QA set with exact-match / F1 and a text-RAG baseline

## Repository structure

```
Dataset/
  HypeRelDocRED/train/     enriched dataset versions (JSONL, Re-DocRED format + qualifiers)
  loadingDataset.py        download Re-DocRED / HyperRED from Hugging Face, export to JSONL
  creatingNewData*.py      LLM enrichment scripts (successive iterations; v3 script = v6 pipeline)
  postProcess.py           rule-based cleaning
src/
  KGConstruction/          load original and hyper-relational graphs into Neo4j
  buildKG/                 render document graphs as HTML
  evalDataset/             dataset-quality judge, QA generation, GraphRAG comparison
KGs/, results/             example graph visualisations and evaluation outputs
docs/                      figures for this README
*.xlsx                     per-run evaluation results (timestamped)
```

## Running it

Requirements: Python 3.10+, a Neo4j instance, and Gemini API keys.

```bash
pip install datasets google-genai pydantic python-dotenv pandas openpyxl matplotlib langchain-community neo4j
```

1. **Get the source data.** The original datasets are not redistributed here. Download them from Hugging Face with `Dataset/loadingDataset.py` (`tonytan48/Re-DocRED`, `declare-lab/HyperRED`).
2. **Create a `.env`** (git-ignored):
   ```
   GEMINI_API_KEY=...
   GEMINI_API_KEY1=...        # the enrichment scripts rotate through GEMINI_API_KEY1..N
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=...
   ```
3. **Build the graphs** with the scripts in `src/KGConstruction/`. `graphRAG.py` expects two Neo4j databases named `original` and `hyper`; the construction scripts write to the default database, so set `Neo4jGraph(database=...)` accordingly before running each one.
4. **Evaluate** with the scripts in `src/evalDataset/`.

File paths are set as constants in each script; adjust them to the version you want to use.

## Data sources

- **Re-DocRED** — Tan et al., *Revisiting DocRED: Addressing the False Negative Problem in Relation Extraction*, EMNLP 2022. [GitHub](https://github.com/tonytan48/Re-DocRED)
- **HyperRED** — Chia et al., *A Dataset for Hyper-Relational Extraction and a Cube-Filling Approach*, EMNLP 2022. [GitHub](https://github.com/declare-lab/HyperRED)
- Relation IDs follow [Wikidata](https://www.wikidata.org/) properties.

HypeRelDocRED is derived from Re-DocRED; use of the derived data is subject to the original dataset's license.
