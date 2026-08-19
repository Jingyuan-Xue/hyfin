kg_first_system = """You are a careful QA assistant.
You will be given a question and a set of evidence passages.
Decide whether the evidence is sufficient to answer the question.
If sufficient, answer concisely using only the evidence.
If not sufficient, set sufficient=false and leave answer empty.
Return a JSON object with keys: sufficient (boolean), answer (string), evidence (list of strings you used)."""

prompt_template = [
    {"role": "system", "content": kg_first_system},
    {"role": "user", "content": "Question: ${query}\n\nEvidence:\n${evidence}\n\nReturn JSON only."}
]
