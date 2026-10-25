import json

with open("rag_autogen_comparison.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        new_source = []
        for line in cell["source"]:
            # Fix any replacement characters with proper em-dash
            fixed_line = line.replace("\ufffd", "—")
            new_source.append(fixed_line)
        cell["source"] = new_source

with open("rag_autogen_comparison.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed character encodings in notebook headings!")
