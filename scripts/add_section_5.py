from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 5: Define the Mock LLM (simulates GPT-4 behavior)
sec5_md = """# Section 5: Define the Mock LLM (simulates GPT-4 behavior)

In this section, we create a deterministic **Mock LLM**. 
Real cloud APIs (like OpenAI GPT-4 or Anthropic Claude) charge fees per token, require active network connections, and can suffer from rate-limiting or quota exhaustion. To make this notebook 100% free, offline, and reproducibly evaluable by teachers and students, we simulate LLM behavior with a dedicated Python function:

- **Mode 1 (WITHOUT RAG — System 1):** The LLM relies purely on its "internal parametric weights". Because the model has no external reference document, it outputs answers containing realistic hallucinations—such as dangerously inflated drug dosages, wrong first-line therapies, or trivializing serious clinical symptoms.
- **Mode 2 (WITH RAG — System 2):** The LLM receives the retrieved text chunks inside its prompt context. Grounded in this verified medical evidence, it extracts the exact clinical figures (e.g., standard dosages, titration rates, adverse effects, and maximum limits) and generates an accurate, high-fidelity response.
"""
add_markdown_cell(nb, sec5_md)

sec5_code = """# Define a Mock LLM generator function that simulates GPT-4 with and without retrieval grounding
def mock_llm_generate(prompt_question, retrieved_context=None):
    # Normalize the query string to lowercase for reliable keyword routing
    clean_query = prompt_question.lower()
    
    # -------------------------------------------------------------------------
    # CASE 1: WITHOUT RAG (Simulating Parametric Memory Hallucination)
    # -------------------------------------------------------------------------
    if retrieved_context is None or len(retrieved_context) == 0:
        if "metformin" in clean_query:
            return (
                "Metformin is usually started at around 100 mg or 250 mg once a day. "
                "Doctors often increase it up to maybe 5000 mg daily depending on the patient. "
                "You can take it whenever you remember, with or without food."
            )
        elif "ibuprofen" in clean_query:
            return (
                "Ibuprofen is a mild painkiller and generally harmless. "
                "Common side effects might be mild drowsiness or dry mouth. "
                "It does not carry serious risks for the stomach or cardiovascular system."
            )
        elif "hypertension" in clean_query:
            return (
                "The first-line treatment for high blood pressure is typically aspirin or generic blood thinners, "
                "along with drinking plenty of herbal tea and avoiding strenuous exercise."
            )
        elif "appendicitis" in clean_query:
            return (
                "Appendicitis causes generalized mild back pain and itchy skin. "
                "The discomfort typically resolves on its own within a few days with simple bed rest."
            )
        elif "paracetamol" in clean_query:
            return (
                "Adults can safely take up to 8000 mg to 10000 mg of Paracetamol per day if they have a bad headache, "
                "because Paracetamol is completely non-toxic and has no harmful effects on liver function."
            )
        else:
            return "This condition can be treated with standard over-the-counter vitamins and plenty of rest."

    # -------------------------------------------------------------------------
    # CASE 2: WITH RAG (Simulating Grounded Context Synthesis)
    # -------------------------------------------------------------------------
    else:
        # Combine all retrieved context chunks into a single unified context string
        combined_context = " ".join(retrieved_context).lower()
        
        if "metformin" in clean_query and "metformin" in combined_context:
            return (
                "Based on the clinical guidelines, the recommended starting dosage of Metformin for Type 2 diabetes "
                "in adults is 500 mg once or twice daily, or 850 mg once daily, taken with meals to minimize gastrointestinal upset. "
                "Dosage may be titrated upward by 500 mg weekly or 850 mg every two weeks. "
                "The maintenance dosage is 1500 mg to 2000 mg per day in divided doses, with a maximum safe dosage of 2550 mg per day."
            )
        elif "ibuprofen" in clean_query and "ibuprofen" in combined_context:
            return (
                "According to the medical documentation, common side effects of Ibuprofen include gastrointestinal distress, "
                "nausea, dyspepsia, heartburn, gastric ulceration, and potential gastrointestinal bleeding. "
                "Prolonged or high-dose therapy increases the risk of renal impairment, fluid retention, hypertension, "
                "and serious cardiovascular thrombotic events including myocardial infarction."
            )
        elif "hypertension" in clean_query and "hypertension" in combined_context:
            return (
                "Grounded in clinical management guidelines, first-line pharmacological treatment for primary hypertension "
                "consists of four major drug classes: Thiazide diuretics (e.g., Chlorthalidone, Hydrochlorothiazide), "
                "ACE inhibitors (e.g., Lisinopril), Angiotensin Receptor Blockers (ARBs, e.g., Losartan), "
                "and Calcium Channel Blockers (CCBs, e.g., Amlodipine). This is paired with lifestyle modifications "
                "including a low-sodium DASH diet, regular aerobic exercise, and weight reduction."
            )
        elif "appendicitis" in clean_query and "appendicitis" in combined_context:
            return (
                "Clinical evidence indicates that acute appendicitis classically begins with dull periumbilical pain "
                "(around the navel) that migrates to the right lower quadrant (McBurney's point). "
                "Associated clinical symptoms include nausea, vomiting, loss of appetite (anorexia), low-grade fever, "
                "abdominal guarding, and localized rebound tenderness."
            )
        elif "paracetamol" in clean_query and "paracetamol" in combined_context:
            return (
                "Based on verified pharmacological guidelines, the standard single dose of Paracetamol for healthy adults "
                "is 500 mg to 1000 mg every 4 to 6 hours as needed. The absolute safe maximum dosage is 4000 mg (4 grams) "
                "per 24-hour day from all combined sources. Exceeding 4000 mg per day significantly increases the risk "
                "of severe acute hepatotoxicity and irreversible liver failure."
            )
        else:
            return f"Retrieved medical context provided: {retrieved_context[0][:150]}..."

# Test the Mock LLM with a query in both modes to demonstrate the hallucination vs. grounded contrast
test_question = "What is the safe dosage of Paracetamol per day for adults?"

print("=" * 80)
print(f"TEST QUESTION: '{test_question}'")
print("=" * 80)

# Simulate System 1 generation (no context)
unassisted_answer = mock_llm_generate(test_question, retrieved_context=None)
print("[SYSTEM 1 - WITHOUT RAG (Hallucinated Answer)]:")
print(unassisted_answer)
print("-" * 80)

# Simulate System 2 generation (with retrieved context)
retrieved_docs, _ = retrieve_relevant_chunks(test_question, top_k=1)
rag_grounded_answer = mock_llm_generate(test_question, retrieved_context=retrieved_docs)
print("[SYSTEM 2 - WITH RAG (Grounded Answer)]:")
print(rag_grounded_answer)
print("=" * 80)
"""
add_code_cell(nb, sec5_code)

save_notebook(nb)
print("Section 5 appended successfully. Total cells:", len(nb["cells"]))
