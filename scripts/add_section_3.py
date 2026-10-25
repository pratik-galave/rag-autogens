from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 3: Create a Domain-Specific Knowledge Base (for RAG)
sec3_md = """# Section 3: Create a Domain-Specific Knowledge Base (for RAG)

In this section, we construct a verified, clinical medical knowledge base. 
In a real-world enterprise setting, this knowledge base might consist of thousands of clinical practice guidelines, FDA drug inserts, or PubMed research papers. Here, we define a focused, high-precision corpus containing the exact clinical ground truth for our 5 evaluation questions, alongside adjacent medical facts to demonstrate semantic discrimination.

### 📚 Target Medical Topics
1. **Metformin Dosage:** Initial doses (500mg - 850mg), titration, and maximum daily limit (2550mg).
2. **Ibuprofen Adverse Effects:** GI ulcers, bleeding, nausea, renal impairment, and cardiovascular risks.
3. **Hypertension First-Line Therapy:** Thiazide diuretics, ACE inhibitors, ARBs, and Calcium Channel Blockers.
4. **Appendicitis Clinical Presentation:** Periumbilical pain migrating to right lower quadrant (RLQ / McBurney's point), nausea, anorexia, fever.
5. **Paracetamol Safe Thresholds:** 500-1000mg per dose, strict 4000mg (4g) daily maximum limit for adults, risk of hepatotoxicity.
"""
add_markdown_cell(nb, sec3_md)

sec3_code = """# Define a domain-specific clinical knowledge base as a list of curated factual text chunks
medical_knowledge_base = [
    # Chunk 1: Metformin dosing guidelines for Type 2 diabetes
    (
        "Metformin is the first-line oral medication for Type 2 diabetes mellitus. "
        "The standard initial dosage for adults is 500 mg once or twice daily, or 850 mg once daily, "
        "taken with meals to minimize gastrointestinal upset. The dosage can be titrated upward "
        "gradually by 500 mg weekly or 850 mg every two weeks based on glycemic response. "
        "The common maintenance dosage ranges from 1500 mg to 2000 mg per day in divided doses. "
        "The maximum recommended safe dosage of Metformin is 2550 mg per day in adults."
    ),
    
    # Chunk 2: Ibuprofen pharmacology and adverse side effects
    (
        "Ibuprofen is a widely used nonsteroidal anti-inflammatory drug (NSAID) indicated for mild to moderate pain and fever. "
        "Common adverse effects include gastrointestinal distress, nausea, dyspepsia, heartburn, gastric ulceration, "
        "and potential gastrointestinal bleeding. Prolonged high-dose therapy is associated with increased risk of renal impairment, "
        "fluid retention, exacerbation of hypertension, and elevated cardiovascular thrombotic risks including myocardial infarction."
    ),
    
    # Chunk 3: Clinical management guidelines for primary hypertension
    (
        "First-line pharmacological treatment for primary hypertension in non-black adult patients consists of four major drug classes: "
        "Thiazide diuretics (such as Chlorthalidone or Hydrochlorothiazide), Angiotensin-Converting Enzyme (ACE) inhibitors (such as Lisinopril), "
        "Angiotensin Receptor Blockers (ARBs, such as Losartan), and Calcium Channel Blockers (CCBs, such as Amlodipine). "
        "Treatment is combined with non-pharmacological lifestyle modifications including a low-sodium DASH diet, exercise, and weight reduction."
    ),
    
    # Chunk 4: Clinical symptoms and diagnostic signs of acute appendicitis
    (
        "Acute appendicitis is a medical emergency caused by inflammation of the vermiform appendix. "
        "The hallmark presentation begins with dull, poorly localized abdominal pain around the periumbilical region (navel) "
        "that subsequently migrates over 12 to 24 hours to the right lower quadrant (McBurney's point). "
        "Accompanying classical symptoms include nausea, vomiting, loss of appetite (anorexia), low-grade fever, "
        "abdominal guarding, and localized rebound tenderness."
    ),
    
    # Chunk 5: Safe adult dosing and toxicity limits for Paracetamol (Acetaminophen)
    (
        "Paracetamol (Acetaminophen) is a first-line analgesic and antipyretic drug. "
        "For healthy adults, the standard recommended single dose is 500 mg to 1000 mg administered every 4 to 6 hours as needed. "
        "The absolute safe maximum dosage of Paracetamol is 4000 mg (4 grams) per 24-hour day from all combined sources. "
        "Exceeding 4000 mg per day significantly increases the risk of severe acute hepatotoxicity, irreversible liver failure, and death. "
        "In patients with chronic liver disease or alcohol misuse, the recommended ceiling is reduced to 2000 mg to 3000 mg per day."
    ),
    
    # Chunk 6: Supplementary medical context — Asthma management (distractor chunk for realistic vector search)
    (
        "Asthma is a chronic inflammatory disorder of the airways. First-line relief therapy includes inhaled short-acting "
        "beta-2 agonists (SABA such as Albuterol). Maintenance therapy relies on inhaled corticosteroids (ICS such as Fluticasone) "
        "to prevent exacerbations and bronchospasms."
    ),
    
    # Chunk 7: Supplementary medical context — Type 1 Diabetes and Insulin (distractor chunk)
    (
        "Type 1 diabetes is an autoimmune condition resulting in absolute insulin deficiency. Management requires lifelong basal-bolus "
        "insulin therapy (such as Insulin Glargine for basal coverage and Insulin Lispro for mealtime coverage) with continuous glucose monitoring."
    )
]

# Display the total number of knowledge base passages indexed
print(f" Successfully loaded {len(medical_knowledge_base)} medical knowledge chunks into memory.")

# Print a preview of the first medical chunk to verify structure and clarity
print("-" * 75)
print("Preview of Knowledge Base Chunk 1 (Metformin):")
print(medical_knowledge_base[0][:210] + "...")
print("-" * 75)
"""
add_code_cell(nb, sec3_code)

save_notebook(nb)
print("Section 3 appended successfully. Total cells:", len(nb["cells"]))
