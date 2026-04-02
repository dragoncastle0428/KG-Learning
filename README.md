# 🕸️ Skill-Driven Hypergraph Knowledge Extraction

This repository explores the implementation of an advanced, reflective Knowledge Graph (KG) extraction pipeline. Moving beyond simple static zero-shot prompting, this project implements a **Skill-Driven Hypergraph Generation** framework (inspired by the *Hyper-KGGen* methodology) using Large Language Models.

The core objective is to extract complex, N-ary relationships (Hyperedges) from unstructured text and continuously improve the extraction accuracy through an autonomous reflection and memory management system (Experience Pool).

## 🎯 Research Objectives
1.  **Hypergraph Reification:** Model complex multi-entity interactions as central 'Event' nodes rather than lossy binary edges.
2.  **Skill-Guided Extraction:** Utilize an dynamic 'Experience Pool' to guide the LLM in identifying intricate relation triggers and anchor cues.
3.  **Self-Reflection Loop:** Implement a feedback mechanism where the LLM reflects on successful or missed extractions to generate new, reusable extraction skills.

## 🧠 System Architecture & Prompting Pipeline

The pipeline consists of four distinct LLM agents, each utilizing specific prompt templates:

1.  **Entity Extractor**
    * *Role:* Identifies all distinct Named Entities within the text.
    * *Output:* A strict JSON list of entities with `name`, `type`, and `description`.
2.  **Relation Extractor (Hyperedge Builder)**
    * *Role:* Constructs N-ary relationships (Event/Hyperedge -> Roles -> Entities).
    * *Mechanism:* Uses dynamically injected *Experiences* from the memory pool to guide the extraction strategy (e.g., identifying temporal, spatial, or conditional qualifiers).
3.  **Reflector (Unstable/Missed Set Analysis)**
    * *Role:* Analyzes hard cases or past extraction trajectories to distill **ONE** highly reusable skill (Trigger + Action).
    * *Constraint:* Avoids entity names; uses abstract role placeholders and discourse patterns.
4.  **Skill Manager (Experience Pool Updater)**
    * *Role:* Maintains the quality of the Experience Pool.
    * *Actions:* Evaluates newly generated skills and decides to `ADD`, `MERGE`, `SKIP`, or `DELETE` based on redundancy and impact, returning operations in JSON format.

## 🛠️ Tech Stack
* **LLM Engine:** `Google Gemini 1.5 Flash/Pro` (Optimized for long context and complex JSON structured outputs)
* **Parsing & Validation:** `Pydantic` (Strict schema enforcement for Entity and Hyperedge structures)
* **Graph Database:** `Neo4j` (Cypher queries for reified hypergraph storage)
* **Environment Management:** `python-dotenv` (Secure API key handling)