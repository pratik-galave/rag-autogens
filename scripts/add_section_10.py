from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 10: Comparison — Graphs and Visualizations
sec10_md = """# Section 10: Comparison — Graphs and Visualizations

In this section, we generate **4 publication-quality visualizations** to visually communicate the benchmark findings:
- **Graph 1: Bar Chart — Faithfulness Scores (System 1 vs. System 2):** Side-by-side grouped bars for each of the 5 questions, illustrating the severe penalty caused by ungrounded hallucinations.
- **Graph 2: Bar Chart — Completeness Scores (System 1 vs. System 2):** Side-by-side grouped bars demonstrating how RAG captures the full breadth of necessary clinical criteria.
- **Graph 3: Bar Chart — Overall Average Score Comparison:** Macroscopic performance summary comparing overall mean scores for both systems.
- **Graph 4: Table/Heatmap — Hallucination Flag Comparison:** A binary safety matrix where **Red = Hallucination Detected (Risk)** and **Green = Grounded & Factual (Safe)**.

> **Color Palette:** System 1 (No RAG) is styled in warm red/coral (`#E63946`), and System 2 (With RAG) is styled in cool teal/emerald (`#2A9D8F`).
"""
add_markdown_cell(nb, sec10_md)

sec10_code = """# Define question labels for clear horizontal axis labeling
question_labels = [f"Q{i}: " + q[:28] + "..." for i, q in enumerate(df_comparison["Question"], start=1)]
x_indices = np.arange(len(question_labels))
bar_width = 0.35

# Set distinct brand colors: Red/Coral for System 1 (Without RAG), Emerald/Teal for System 2 (With RAG)
color_sys1 = "#E63946"  # Vibrant Coral Red (System 1)
color_sys2 = "#2A9D8F"  # Vibrant Emerald Teal (System 2)

# =============================================================================
# GRAPH 1: Bar Chart — Faithfulness Scores: System 1 vs System 2 (Side by Side)
# =============================================================================
plt.figure(figsize=(12, 6), dpi=120)

# Create bars for System 1 faithfulness
bars_s1_faith = plt.bar(
    x_indices - bar_width/2, 
    df_comparison["Sys1 Faithfulness"], 
    width=bar_width, 
    label="System 1: Standard AutoGen (WITHOUT RAG)", 
    color=color_sys1, 
    edgecolor="black", 
    alpha=0.9
)

# Create bars for System 2 faithfulness
bars_s2_faith = plt.bar(
    x_indices + bar_width/2, 
    df_comparison["Sys2 Faithfulness"], 
    width=bar_width, 
    label="System 2: RAG-Enhanced AutoGen (WITH RAG)", 
    color=color_sys2, 
    edgecolor="black", 
    alpha=0.9
)

# Add numeric value labels on top of each bar for readability
for bar in bars_s1_faith:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

for bar in bars_s2_faith:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Configure chart aesthetics, titles, and grid
plt.title("Graph 1: Faithfulness Score Comparison per Question (System 1 vs System 2)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Evaluation Questions", fontsize=12, labelpad=10)
plt.ylabel("Faithfulness Score (0.0 to 1.0)", fontsize=12, labelpad=10)
plt.xticks(x_indices, question_labels, rotation=15, ha="right", fontsize=10)
plt.ylim(0, 1.15)
plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=11)
plt.tight_layout()
plt.show()

# =============================================================================
# GRAPH 2: Bar Chart — Completeness Scores: System 1 vs System 2 (Side by Side)
# =============================================================================
plt.figure(figsize=(12, 6), dpi=120)

# Create bars for System 1 completeness
bars_s1_comp = plt.bar(
    x_indices - bar_width/2, 
    df_comparison["Sys1 Complete"], 
    width=bar_width, 
    label="System 1: Standard AutoGen (WITHOUT RAG)", 
    color=color_sys1, 
    edgecolor="black", 
    alpha=0.9
)

# Create bars for System 2 completeness
bars_s2_comp = plt.bar(
    x_indices + bar_width/2, 
    df_comparison["Sys2 Complete"], 
    width=bar_width, 
    label="System 2: RAG-Enhanced AutoGen (WITH RAG)", 
    color=color_sys2, 
    edgecolor="black", 
    alpha=0.9
)

# Add numeric value labels on top of each bar
for bar in bars_s1_comp:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

for bar in bars_s2_comp:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Configure chart aesthetics, titles, and grid
plt.title("Graph 2: Completeness Score Comparison per Question (System 1 vs System 2)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Evaluation Questions", fontsize=12, labelpad=10)
plt.ylabel("Completeness Score (0.0 to 1.0)", fontsize=12, labelpad=10)
plt.xticks(x_indices, question_labels, rotation=15, ha="right", fontsize=10)
plt.ylim(0, 1.15)
plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=11)
plt.tight_layout()
plt.show()

# =============================================================================
# GRAPH 3: Bar Chart — Overall Average Score Comparison (Single Bar Each)
# =============================================================================
plt.figure(figsize=(8, 6), dpi=120)

# Compute the mean overall score (average of faithfulness and completeness combined) for each system
mean_overall_sys1 = (avg_faith_sys1 + avg_comp_sys1) / 2.0
mean_overall_sys2 = (avg_faith_sys2 + avg_comp_sys2) / 2.0

system_labels = ["System 1\\n(WITHOUT RAG)", "System 2\\n(WITH RAG)"]
mean_scores = [mean_overall_sys1, mean_overall_sys2]
bar_colors = [color_sys1, color_sys2]

# Draw single summary bars for each system
summary_bars = plt.bar(system_labels, mean_scores, width=0.45, color=bar_colors, edgecolor="black", alpha=0.9)

# Add text labels displaying the exact overall average score on top of each bar
for bar in summary_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.03, f"{height:.3f} ({height*100:.1f}%)", ha="center", va="bottom", fontsize=12, fontweight="bold")

# Configure summary chart labels and styling
plt.title("Graph 3: Overall Average Benchmark Score (System 1 vs System 2)", fontsize=14, fontweight="bold", pad=15)
plt.ylabel("Combined Average Score (0.0 to 1.0)", fontsize=12, labelpad=10)
plt.ylim(0, 1.20)
plt.axhline(1.0, color="#888888", linestyle="--", linewidth=1, label="Perfect Benchmark (1.0)")
plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="upper left", fontsize=11)
plt.tight_layout()
plt.show()

# =============================================================================
# GRAPH 4: Table/Heatmap — Hallucination Flag Comparison (True=Red, False=Green)
# =============================================================================
plt.figure(figsize=(9, 6), dpi=120)

# Construct a numerical matrix where 1 represents Hallucination (True) and 0 represents Safe / Grounded (False)
heatmap_matrix = np.array([
    [1 if val else 0 for val in df_comparison["Sys1 Hallucinated"]],
    [1 if val else 0 for val in df_comparison["Sys2 Hallucinated"]]
]).T  # Transpose so rows are questions and columns are systems

# Create custom text annotations for each cell
cell_annotations = []
for row in heatmap_matrix:
    row_text = []
    for val in row:
        if val == 1:
            row_text.append("HALLUCINATED\\n(RISK)")
        else:
            row_text.append("GROUNDED\\n(SAFE)")
    cell_annotations.append(row_text)

# Define custom colormap: Green (#2A9D8F) for 0 (Safe) and Red (#E63946) for 1 (Hallucination)
custom_safety_cmap = sns.color_palette([color_sys2, color_sys1])

# Generate the heatmap
sns.heatmap(
    heatmap_matrix, 
    annot=np.array(cell_annotations), 
    fmt="", 
    cmap=custom_safety_cmap, 
    cbar=False, 
    linewidths=2.5, 
    linecolor="white",
    xticklabels=["System 1 (No RAG)", "System 2 (With RAG)"],
    yticklabels=[f"Q{i}: " + q[:30] + "..." for i, q in enumerate(df_comparison['Question'], start=1)],
    annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"}
)

# Configure heatmap title and layout
plt.title("Graph 4: Hallucination Risk Heatmap (Red = Dangerous Hallucination, Green = Clinically Safe)", fontsize=13, fontweight="bold", pad=15)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
"""
add_code_cell(nb, sec10_code)

save_notebook(nb)
print("Section 10 appended successfully. Total cells:", len(nb["cells"]))
