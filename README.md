# 🕸️ Skill-Driven Hyper-Relational Knowledge Graph Extraction

This repository introduces HyperDocRED, a specialized dataset and framework designed to bridge the gap between document-level reasoning and high-fidelity GraphRAG. By extending Re-DocRED into a hyper-relational format, we focus on extracting not just "what happened," but the full context (when, where, why) using a reflective LLM pipeline.

The core objective is to extract complex relational facts by first identifying **Base Triples (Head, Relation, Tail)**, and then extracting their associated **Qualifiers (Qualifier Relation, Qualifier Value)**. By leveraging an autonomous reflection and memory management system (Experience Pool), the LLM continuously improves its extraction accuracy, especially for subtle or implicit qualifiers.

## 🚀 Project Vision: HyperDocRED
Traditional extraction often loses the nuanced context found in long-form documents. HyperDocRED addresses this by:Hyper-Relational Mapping: Every fact is stored as a Base Triple $(h, r, t)$ with associated Qualifiers $(qr, qv)$.GraphRAG-Ready: Designed specifically to reduce hallucinations and improve retrieval accuracy in complex RAG workflows.

## 📊 Evaluation Framework (The 5 Pillars)
Metric	Description	Goal (100 pts)
1. Hallucination	Measures knowledge fabrication or distortion.	100% grounded in the source text.
2. Comprehensiveness	Richness of details like time, location, and conditions.	Capture every subtle qualifier available.
3. Schema Mapping	Accuracy of mapping to predefined P-IDs (Relations).	Perfect alignment with the target ontology.
4. Salience	Focus on the core themes of the document.	Extracting facts that penetrate the main subject.
5. Entity Clarity	Pronoun resolution and Named Entity merging.	Fully refined/unified entities (No ambiguous "He/She").

## 🎯 Research Objectives
1.  **Hyper-Relational Extraction:** Move beyond traditional binary KGs by strictly separating core factual triples from their contextual qualifiers, aligning perfectly with the HyperRED schema.
2.  **Skill-Guided Qualifier Identification:** Utilize a dynamic 'Experience Pool' to guide the LLM in identifying subtle discourse triggers and anchor cues for specific qualifiers (e.g., temporal boundaries, spatial contexts, conditions).
3.  **Self-Reflection Loop:** Implement a feedback mechanism where the LLM reflects on extraction errors (missed triples or wrong qualifier boundaries) on the HyperRED dataset to generate new, reusable extraction skills.

## 🧠 System Architecture & Prompting Pipeline

The pipeline abandons simple Entity Extraction and focuses entirely on relational structures. It consists of three primary LLM agents:

1.  **Fact Extractor (Base Triple + Qualifiers)**
    * *Role:* Analyzes the text to extract the complete hyper-relational structure.
    * *Step 1:* Extracts the primary Base Triples `(h, r, t)`.
    * *Step 2:* Extracts corresponding Qualifiers `(qr, qv)` that attach ONLY to the identified base triples.
    * *Mechanism:* Uses dynamically injected *Experiences* from the memory pool to guide the extraction strategy for tricky relations or implicit qualifiers.
2.  **Reflector (Error Analysis Agent)**
    * *Role:* Compares the LLM's predicted Relational Facts against the HyperRED Ground Truth. Analyzes hard cases (e.g., "Why was this temporal qualifier missed?") to distill **ONE** highly reusable skill (Trigger + Action).
    * *Constraint:* Avoids specific entity names; uses abstract role placeholders and discourse patterns to ensure generalization.
3.  **Skill Manager (Experience Pool Updater)**
    * *Role:* Maintains the quality of the Experience Pool.
    * *Actions:* Evaluates newly generated skills and decides to `ADD`, `MERGE`, `SKIP`, or `DELETE` based on redundancy and impact.

## 🛠️ Tech Stack
* **LLM Engine:** `Google Gemini` (Optimized for long context and complex JSON structured outputs)
* **Parsing & Validation:** `Pydantic` (Strict schema enforcement for Triples and Qualifiers)
* **Data Processing:** `Hugging Face Datasets` & `Pandas` (For HyperRED ingestion and manipulation)
* **Graph Database:** `Neo4j` (Cypher queries for LPG-based hyper-relational storage)

## 📈 Research Objectives
Benchmarking: Comparing HyperDocRED against standard Re-DocRED to prove the utility of qualifiers.
Noise Reduction: Minimizing Hallucination (Metric 1) while maximizing Comprehensiveness (Metric 2).
Graph-to-Text: Evaluating how well the extracted KG can reconstruct the original document's core message (Salience).