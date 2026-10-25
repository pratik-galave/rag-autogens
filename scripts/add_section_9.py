from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 9: Comparison — Numerical Scores and Analysis
sec9_md = """# Section 9: Comparison — Numerical Scores and Analysis

In this section, we consolidate the empirical evaluation results into structured tables using `pandas`:
1. **Granular Comparison Table:** A clean DataFrame displaying per-question metrics:
   - `Question`
   - `Sys1 Faithfulness` vs. `Sys2 Faithfulness`
   - `Sys1 Complete` vs. `Sys2 Complete`
   - `Sys1 Hallucinated` vs. `Sys2 Hallucinated`
2. **Aggregate Benchmark Statistics:** Summary calculations computing average faithfulness, average completeness, and hallucination rates for both architectures.
3. **Qualitative Textual Comparison:** A side-by-side inspection of generated answers, highlighting specific clinical risks and hallucinations observed in System 1 versus grounded corrections in System 2.
"""
add_markdown_cell(nb, sec9_md)

sec9_code = """# Build the primary comparison data dictionary matching the exact required schema
comparison_data = []

for s1, s2 in zip(system1_evaluation_records, system2_evaluation_records):
    comparison_data.append({
        "Question": s1["question"],
        "Sys1 Faithfulness": s1["faithfulness"],
        "Sys2 Faithfulness": s2["faithfulness"],
        "Sys1 Complete": s1["completeness"],
        "Sys2 Complete": s2["completeness"],
        "Sys1 Hallucinated": s1["hallucinated"],
        "Sys2 Hallucinated": s2["hallucinated"]
    })

# Convert the list of records into a clean pandas DataFrame
df_comparison = pd.DataFrame(comparison_data)

# Display the main comparison table cleanly
print("=" * 115)
print("                                 PRIMARY SYSTEM COMPARISON SUMMARY TABLE")
print("=" * 115)
display(df_comparison)
print("=" * 115)

# Calculate macroscopic aggregate statistics for System 1 (Without RAG)
avg_faith_sys1 = df_comparison["Sys1 Faithfulness"].mean()
avg_comp_sys1 = df_comparison["Sys1 Complete"].mean()
hallucination_rate_sys1 = (df_comparison["Sys1 Hallucinated"].sum() / len(df_comparison)) * 100

# Calculate macroscopic aggregate statistics for System 2 (With RAG)
avg_faith_sys2 = df_comparison["Sys2 Faithfulness"].mean()
avg_comp_sys2 = df_comparison["Sys2 Complete"].mean()
hallucination_rate_sys2 = (df_comparison["Sys2 Hallucinated"].sum() / len(df_comparison)) * 100

# Construct an aggregate summary DataFrame to highlight macroscopic architectural gains
aggregate_summary_df = pd.DataFrame({
    "System Architecture": ["System 1 (Standard AutoGen, No RAG)", "System 2 (RAG-Enhanced AutoGen, With RAG)"],
    "Mean Faithfulness Score": [round(avg_faith_sys1, 3), round(avg_faith_sys2, 3)],
    "Mean Completeness Score": [round(avg_comp_sys1, 3), round(avg_comp_sys2, 3)],
    "Hallucination Rate (%)": [f"{hallucination_rate_sys1:.1f}%", f"{hallucination_rate_sys2:.1f}%"],
    "Factual Grounding Mechanism": ["Parametric Memory Only (No external facts)", "FAISS Dense Retrieval (all-MiniLM-L6-v2)"]
})

print("\\n" + "=" * 95)
print("                            OVERALL MACROSCOPIC PERFORMANCE COMPARISON")
print("=" * 95)
display(aggregate_summary_df)
print("=" * 95)

# Print a qualitative side-by-side textual analysis for all 5 questions
print("\\n" + "=" * 105)
print("                      QUALITATIVE ANSWER COMPARISON: CLINICAL ACCURACY AUDIT")
print("=" * 105)

for idx, (s1, s2) in enumerate(zip(system1_evaluation_records, system2_evaluation_records), start=1):
    print(f"\\n--- QUESTION {idx}: {s1['question']} ---")
    print(f"❌ [SYSTEM 1 - NO RAG]: {s1['answer']}")
    print(f"   -> Faithfulness: {s1['faithfulness']} | Completeness: {s1['completeness']} | Hallucination Detected: {s1['hallucinated']}")
    print(f"✅ [SYSTEM 2 - WITH RAG]: {s2['answer']}")
    print(f"   -> Faithfulness: {s2['faithfulness']} | Completeness: {s2['completeness']} | Hallucination Detected: {s2['hallucinated']}")
    print("-" * 105)
"""
add_code_cell(nb, sec9_code)

save_notebook(nb)
print("Section 9 appended successfully. Total cells:", len(nb["cells"]))
