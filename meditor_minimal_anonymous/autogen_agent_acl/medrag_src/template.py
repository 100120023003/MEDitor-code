from liquid import Template



from jinja2 import Template



general_cot_system ='''You are a helpful medical expert, and your task is to answer a multi-choice medical question.
First, think step-by-step to reach a conclusion.
Then you MUST output a single valid JSON object in the following format:

Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/D}}

Requirements:
- "step_by_step_thinking" should be a SHORT explanation (no more than 5 sentences).
- "answer_choice" MUST be exactly one of "A", "B", "C", "D" (capital letter only).
- You MUST output ONLY the JSON object, with no extra text, no markdown, no comments before or after.
Your responses will be used for research purposes only, so please always give a definite answer.'''



general_cot =Template ('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step (no more than 5 sentences),
and then output ONLY ONE LINE of valid JSON in the following format:

{"step_by_step_thinking": "...", "answer_choice": "A"}
''')





general_medrag_system ='''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents.
First, carefully read the provided documents, but treat them as noisy evidence that may be partially relevant or off-topic.
Then think step-by-step to reach a conclusion.
Finally you MUST output a single valid JSON object in the following format:

Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/D}}

Requirements:
- "step_by_step_thinking" should be a SHORT explanation (no more than 5 sentences), using the documents when they are relevant.
- "answer_choice" MUST be exactly one of "A", "B", "C", "D" (capital letter only).
- If the documents are weak, conflicting, or off-topic, rely on the question and your medical knowledge instead of refusing to answer.
- Do NOT answer with "None of the above" or any free-text choice unless that exact option appears in the provided choices.
- You MUST output ONLY the JSON object, with no extra text, no markdown, no comments before or after.
Your responses will be used for research purposes only, so please always give a definite answer.'''



general_medrag =Template ('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please:
1) Read the above documents, but ignore any snippet that is irrelevant or contradictory.
2) If the documents are weak or off-topic, answer from the question and your medical knowledge.
3) Think step-by-step (no more than 5 sentences).
4) Output ONLY ONE LINE of valid JSON in the following format:

{"step_by_step_thinking": "...", "answer_choice": "A"}
''')



meditron_medrag =Template ('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')



simple_medrag_system ='''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents.'''

simple_medrag_prompt =Template ('''Here are the relevant documents:\n{{context}}\nHere is the question:\n{{question}}''')



i_medrag_system ='''You are a helpful medical assistant, and your task is to answer the given question following the instructions given by the user. '''



follow_up_instruction_ask ='''Please first analyze all the information in a section named Analysis (## Analysis). Then, use key terms from previous answers to form specific and direct questions. Generate {} concise, context-specific queries to search for additional information in an external knowledge base, in a section named Queries (## Queries). Each query should be simple and focused, directly relating to the key terms used in the answers. Wait for responses from the user before proceeding.'''

follow_up_instruction_answer ='''Please first think step-by-step to analyze all the information in a section named Analysis (## Analysis). Then, please provide your answer choice in a section named Answer (## Answer).'''

