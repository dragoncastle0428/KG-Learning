JOINT_EXTRACTION_PROMPT = """
You are an expert Hyper-Relational Fact Extraction assistant. 
Your task is to analyze the given text and JOINTLY extract the core relational facts along with their associated qualifiers.

Text: {raw_text}

**Experiences (Skill Pool):**
The following are reference patterns and past mistakes learned from previous extraction tasks. Use these experiences to guide your extraction, paying special attention to tricky relations and subtle qualifiers.
{experiences}

**Extraction Rules (CRITICAL):**
1. **Identify Base Triples:** Find the core relational facts in the format of (head, relation, tail).
2. **Identify Qualifiers:** For each base triple, find any conditional, temporal, spatial, or descriptive attributes that specifically modify that relation.
3. **EXACT MATCH ONLY:** The values for 'head', 'tail', and 'qualifier_value' MUST be exact contiguous substrings copied directly from the original Text. Do not rephrase, translate, or normalize the text. If you change even one character, the downstream span mapping will fail.
4. If the text contains no meaningful relational facts, output an empty list for "facts".
5. Output MUST be valid JSON only, following the exact schema below. Do not include markdown formatting blocks (like ```json).

**JSON Output Schema:**
{{
  "facts": [
    {{
      "head": "Exact text of the subject entity",
      "relation": "The type of relation",
      "tail": "Exact text of the object entity",
      "qualifiers": [
        {{
          "qualifier_relation": "The type of qualifier",
          "qualifier_value": "Exact text of the qualifier"
        }}
      ]
    }}
  ]z
}}
"""