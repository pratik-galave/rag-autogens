import json

with open("rag_autogen_comparison.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\nMarkdown Section Headings:")
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown":
        first_line = cell["source"][0].strip()
        print(f"Cell {i:2d} (MD): {first_line}")
    elif cell["cell_type"] == "code":
        num_outputs = len(cell.get("outputs", []))
        print(f"Cell {i:2d} (Code): {num_outputs} output items captured")
