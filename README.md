🕸️ Skill-Driven Hyper-Relational KG & GraphRAG Framework
This repository focuses on building an advanced, reflective pipeline for Document-level Hyper-Relational Knowledge Graph (HKG) Construction. By evolving from the HyperRED dataset to the newly proposed HyperDocRED, this project aims to bridge the gap between complex relational extraction and high-performance GraphRAG applications.

🚀 Project Evolution: From HyperRED to HyperDocRED
While standard hyper-relational extraction focuses on sentence-level facts, real-world knowledge is often scattered across multiple sentences.
HyperDocRED: A specialized dataset derived from Re-DocRED, designed to evaluate Hyper-Relational extraction, Cross-sentence reasoning, and GraphRAG retrieval accuracy.
Vision: Transitioning from "extracting facts" to "constructing queryable, high-fidelity knowledge structures" for LLMs.

🎯 Research Objectives
Hyper-Relational Document Extraction: Extracting complex facts that span across a whole document, strictly separating Base Triples (h, r, t) from their Contextual Qualifiers (qr, qv).
Skill-Guided Reflection: Utilizing a dynamic 'Experience Pool' to teach the LLM how to identify implicit qualifiers and resolve coreferences across document boundaries.
GraphRAG Benchmarking: Evaluating how hyper-relational structures (n-ary relations) reduce hallucinations and improve retrieval precision compared to traditional binary (SPO) graphs.

🧠 System Architecture
The pipeline leverages an autonomous reflection loop to continuously refine its extraction "skills."
1. Fact Extractor (The Practitioner)
Multi-step Inference: Identifies core entities and base triples, then recursively searches for qualifiers (Time, Location, Quantity, etc.) that anchor those facts.
Memory Injection: Dynamically pulls relevant "Experiences" from the pool to handle ambiguous discourse patterns.
2. Reflector (The Analyst)
Ground Truth Comparison: Compares predictions against HyperDocRED labels.
Root Cause Analysis: If a qualifier is missed or a relation is hallucinated, it distills the error into a generalized Skill (Trigger + Action).
3. Skill Manager & Graph Integrator
Experience Pool: Manages the lifecycle of skills (ADD, MERGE, DELETE).
LPG Mapping: Transforms extracted hyper-relations into Labeled Property Graphs (LPG) for Neo4j, ensuring they are optimized for GraphRAG queries.

📊 Dataset & Evaluation
Base Dataset: [HyperDocRED] (Extended from Re-DocRED)
Primary Task: Extracting (h, r, t) with multiple (qr, qv) pairs in a document-level context.
GraphRAG Metrics:
Faithfulness: Does the graph accurately represent the source document?
Retrieval Recall: Can the system answer complex queries that require qualifier-based filtering?

🛠️ Tech Stack
LLM Engine: Google Gemini (Leveraging long-context windows for document-level analysis)
Schema Control: Pydantic (Ensuring JSON outputs strictly follow the hyper-relational schema)
Storage & RAG: Neo4j & LangChain/LlamaIndex (For Cypher-based GraphRAG implementation)
Data Ops: Hugging Face Datasets & Pandas