"""
Helper script to assemble and maintain the Jupyter Notebook (.ipynb)
for RAG vs Standard AutoGen comparison.
"""
import json
import os

NOTEBOOK_PATH = "rag_autogen_comparison.ipynb"

def load_or_create_notebook():
    if os.path.exists(NOTEBOOK_PATH):
        with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 2,
                "pygments_lexer": "ipython3",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def save_notebook(nb):
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def add_markdown_cell(nb, source_text):
    # Ensure source_text is list of lines with \n
    lines = [line + "\n" for line in source_text.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }
    nb["cells"].append(cell)

def add_code_cell(nb, source_code):
    lines = [line + "\n" for line in source_code.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }
    nb["cells"].append(cell)

if __name__ == "__main__":
    nb = load_or_create_notebook()
    save_notebook(nb)
    print("Notebook initialized:", NOTEBOOK_PATH)
