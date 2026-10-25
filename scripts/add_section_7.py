from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 7: System 2 — RAG-Enhanced AutoGen Agent (WITH RAG)
sec7_md = """# Section 7: System 2 — RAG-Enhanced AutoGen Agent (WITH RAG)

In this section, we construct **System 2**: a RAG-enhanced multi-agent architecture.

### 🔬 Multi-Agent Roster for System 2
1. **`OrchestratorAgent`:** Manages conversation dispatch and receives user queries.
2. **`RAGAgent`:** Enhances the generator with dense semantic search:
   - Encodes the user's inquiry into an embedding vector using `SentenceTransformer`.
   - Queries the local `FAISS` vector index to extract the top-$k$ most relevant clinical document chunks.
   - Injects the retrieved chunks into the prompt context to produce a factually grounded answer via `mock_llm_generate(question, retrieved_context)`.
3. **`CriticAgent`:** Rigorously scores the output for Faithfulness, Completeness, and Hallucination using the exact same clinical benchmark criteria as System 1.

This multi-agent flow makes the retrieval, synthesis, and verification steps completely transparent and auditable.
"""
add_markdown_cell(nb, sec7_md)

sec7_code = """# Define the RAGAgent class that combines FAISS retrieval and grounded LLM generation
class RAGAgent:
    def __init__(self, name="RAG AGENT", top_k=2):
        # Store display name and the number of documents to retrieve
        self.name = name
        self.top_k = top_k

    def retrieve_and_generate(self, question, verbose=True):
        if verbose:
            print(f"[{self.name}]    → Searching FAISS vector database for relevant clinical chunks...")
            
        # Retrieve the most relevant text chunks and their L2 similarity scores from FAISS
        retrieved_chunks, distances = retrieve_relevant_chunks(question, top_k=self.top_k)
        
        if verbose:
            print(f"[{self.name}]    → Retrieved {len(retrieved_chunks)} relevant knowledge chunks (Top distance: {distances[0]:.4f})")
            print(f"[{self.name}]    → Synthesizing answer grounded strictly in retrieved context...")
            
        # Generate the answer by passing both the question and the retrieved chunks to the LLM
        grounded_answer = mock_llm_generate(question, retrieved_context=retrieved_chunks)
        
        return grounded_answer, retrieved_chunks

# Define the OrchestratorAgent class to coordinate System 2 multi-agent workflow
class OrchestratorSystem2:
    def __init__(self, top_k=2):
        # Initialize the RAG agent and the evaluation critic agent
        self.rag_agent = RAGAgent(top_k=top_k)
        self.critic = CriticAgent()

    def process_query(self, user_question, verbose=True):
        if verbose:
            print("=" * 85)
            print(f"[ORCHESTRATOR]   → Received question: '{user_question}'")
            print(f"[ORCHESTRATOR]   → Delegating query to {self.rag_agent.name}...")
            
        # Step 1: Delegate query to RAG agent for retrieval and grounded generation
        answer, chunks = self.rag_agent.retrieve_and_generate(user_question, verbose=verbose)
        
        if verbose:
            print(f"[{self.rag_agent.name}]    → Generated answer: {answer}")
            print(f"[ORCHESTRATOR]   → Sending grounded answer to {self.critic.name} for evaluation...")
            
        # Step 2: Critic agent evaluates faithfulness, completeness, and hallucinations
        evaluation_result = self.critic.evaluate(user_question, answer)
        
        if verbose:
            print(
                f"[{self.critic.name}]   → Faithfulness: {evaluation_result['faithfulness']:.2f} | "
                f"Completeness: {evaluation_result['completeness']:.2f} | "
                f"Hallucination Flag: {evaluation_result['hallucinated']}"
            )
            print("=" * 85)
            
        return {
            "question": user_question,
            "answer": answer,
            "retrieved_chunks": chunks,
            "faithfulness": evaluation_result["faithfulness"],
            "completeness": evaluation_result["completeness"],
            "hallucinated": evaluation_result["hallucinated"]
        }

# Demonstrate a sample run of System 2 on Question 1 to verify grounded behavior
system2_orchestrator = OrchestratorSystem2(top_k=2)
sample_q1 = "What is the recommended dosage of Metformin for Type 2 diabetes?"
system2_sample_output = system2_orchestrator.process_query(sample_q1, verbose=True)
"""
add_code_cell(nb, sec7_code)

save_notebook(nb)
print("Section 7 appended successfully. Total cells:", len(nb["cells"]))
