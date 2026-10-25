from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 8: Run Both Systems on the Same Test Questions
sec8_md = """# Section 8: Run Both Systems on the Same Test Questions

In this section, we conduct our controlled empirical benchmark:
- We define the **5 standardized clinical test questions** covering dosages, side effects, first-line treatments, and emergency symptoms.
- We run **System 1 (WITHOUT RAG)** across all 5 questions, allowing the agents to answer using parametric memory alone.
- We run **System 2 (WITH RAG)** across all 5 questions, enabling semantic retrieval of relevant clinical chunks via FAISS before answer synthesis.
- For each question, we print the formatted multi-agent conversational transcript, showing transparently how queries are received, how knowledge is retrieved (in System 2), how answers are generated, and how the `CriticAgent` evaluates each response.
"""
add_markdown_cell(nb, sec8_md)

sec8_code = """# Define the 5 standardized medical evaluation questions
test_questions = [
    "What is the recommended dosage of Metformin for Type 2 diabetes?",
    "What are the side effects of Ibuprofen?",
    "What is the first-line treatment for hypertension?",
    "What are the symptoms of appendicitis?",
    "What is the safe dosage of Paracetamol per day for adults?"
]

# Initialize both multi-agent orchestrator systems
orchestrator_sys1 = OrchestratorSystem1()
orchestrator_sys2 = OrchestratorSystem2(top_k=2)

# Lists to store detailed evaluation records for downstream numerical and visual analysis
system1_evaluation_records = []
system2_evaluation_records = []

print("=" * 90)
print(" RUNNING EXPERIMENTAL BENCHMARK: SYSTEM 1 (STANDARD AUTOGEN WITHOUT RAG)")
print("=" * 90)

# Iterate through each test question and process it through System 1
for idx, question in enumerate(test_questions, start=1):
    print(f"\\n--- [SYSTEM 1: QUESTION {idx}/5] ---")
    result = orchestrator_sys1.process_query(question, verbose=True)
    system1_evaluation_records.append(result)

print("\\n" + "=" * 90)
print(" RUNNING EXPERIMENTAL BENCHMARK: SYSTEM 2 (RAG-ENHANCED AUTOGEN WITH RAG)")
print("=" * 90)

# Iterate through each test question and process it through System 2
for idx, question in enumerate(test_questions, start=1):
    print(f"\\n--- [SYSTEM 2: QUESTION {idx}/5] ---")
    result = orchestrator_sys2.process_query(question, verbose=True)
    system2_evaluation_records.append(result)

print("\\n" + "=" * 90)
print(" Benchmark execution completed successfully for all 5 questions on both systems!")
print("=" * 90)
"""
add_code_cell(nb, sec8_code)

save_notebook(nb)
print("Section 8 appended successfully. Total cells:", len(nb["cells"]))
