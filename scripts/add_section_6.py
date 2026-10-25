from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 6: System 1 — Standard AutoGen Agent (WITHOUT RAG)
sec6_md = """# Section 6: System 1 — Standard AutoGen Agent (WITHOUT RAG)

In this section, we construct **System 1**: a standard multi-agent conversational system simulating Microsoft AutoGen's conversational architecture without RAG.

### 🤖 Multi-Agent Roster for System 1
1. **`OrchestratorAgent`:** Manages conversation state, routes queries to worker agents, and coordinates workflow logging.
2. **`StandardAgent`:** An unassisted generator agent that formulates medical answers relying purely on internal parametric memory (simulated by our mock LLM with `retrieved_context=None`).
3. **`CriticAgent`:** An evaluation agent that inspects the generated answer against clinical benchmarks:
   - **Faithfulness Score (0.0 - 1.0):** Evaluates if the answer contains clinically verified medical terminology and avoids fabricated claims.
   - **Completeness Score (0.0 - 1.0):** Calculates the percentage of required domain keywords present in the response.
   - **Hallucination Flag (`True`/`False`):** Detects whether the answer contains erroneous dosage values, incorrect contraindications, or misleading advice.

All agents communicate by exchanging transparent, structured messages in the standard AutoGen dialogue format.
"""
add_markdown_cell(nb, sec6_md)

sec6_code = """# Define the ground truth clinical evaluation benchmark dictionary
ground_truth_criteria = {
    "metformin": {
        "topic": "Metformin Dosage for Type 2 Diabetes",
        "required_keywords": ["500 mg", "850 mg", "2550 mg", "meals", "titrated"],
        "hallucination_indicators": ["100 mg", "250 mg", "5000 mg", "whenever", "without food"]
    },
    "ibuprofen": {
        "topic": "Side Effects of Ibuprofen",
        "required_keywords": ["gastrointestinal", "ulcer", "bleeding", "renal", "myocardial infarction"],
        "hallucination_indicators": ["harmless", "drowsiness", "dry mouth", "does not carry", "mild painkiller"]
    },
    "hypertension": {
        "topic": "First-Line Treatment for Hypertension",
        "required_keywords": ["thiazide", "ace inhibitors", "arbs", "calcium channel", "lifestyle"],
        "hallucination_indicators": ["aspirin", "blood thinners", "herbal tea", "avoiding strenuous"]
    },
    "appendicitis": {
        "topic": "Symptoms of Appendicitis",
        "required_keywords": ["periumbilical", "right lower quadrant", "mcburney", "anorexia", "rebound tenderness"],
        "hallucination_indicators": ["back pain", "itchy", "resolves on its own", "bed rest"]
    },
    "paracetamol": {
        "topic": "Safe Daily Dosage of Paracetamol for Adults",
        "required_keywords": ["500 mg", "1000 mg", "4000 mg", "hepatotoxicity", "liver"],
        "hallucination_indicators": ["8000 mg", "10000 mg", "completely non-toxic", "no harmful"]
    }
}

# Helper function to match a question to its ground truth criteria key
def get_topic_key(question_text):
    clean_text = question_text.lower()
    for key in ground_truth_criteria:
        if key in clean_text:
            return key
    return None

# Define the CriticAgent class responsible for scoring answers against domain benchmarks
class CriticAgent:
    def __init__(self, name="CRITIC AGENT"):
        # Store the agent's display name
        self.name = name

    def evaluate(self, question, answer):
        # Identify the relevant clinical topic key
        topic_key = get_topic_key(question)
        if not topic_key:
            return {"faithfulness": 0.5, "completeness": 0.5, "hallucinated": False}
        
        criteria = ground_truth_criteria[topic_key]
        answer_lower = answer.lower()
        
        # 1. Check for hallucination flags (detect wrong dosages or misleading terms)
        hallucination_found = any(bad_word in answer_lower for bad_word in criteria["hallucination_indicators"])
        
        # 2. Compute Completeness Score: proportion of expected key terms present
        matched_keywords = [kw for kw in criteria["required_keywords"] if kw in answer_lower]
        completeness_score = round(len(matched_keywords) / len(criteria["required_keywords"]), 2)
        
        # 3. Compute Faithfulness Score: heavily penalized if hallucinations are detected
        if hallucination_found:
            # Low faithfulness when fabricated facts are present
            faithfulness_score = round(max(0.1, 0.35 * completeness_score), 2)
        else:
            # High faithfulness proportional to completeness and absence of false claims
            faithfulness_score = round(0.5 + (0.5 * completeness_score), 2)
            
        return {
            "faithfulness": faithfulness_score,
            "completeness": completeness_score,
            "hallucinated": hallucination_found,
            "matched_keywords": matched_keywords
        }

# Define the StandardAgent class (generates answers from parametric memory without RAG)
class StandardAgent:
    def __init__(self, name="STANDARD AGENT"):
        # Store the agent's display name
        self.name = name

    def generate_response(self, question):
        # Call the mock LLM without any retrieved context
        return mock_llm_generate(question, retrieved_context=None)

# Define the OrchestratorAgent class to coordinate System 1 multi-agent workflow
class OrchestratorSystem1:
    def __init__(self):
        # Initialize the standard generator agent and the critic agent
        self.generator = StandardAgent()
        self.critic = CriticAgent()

    def process_query(self, user_question, verbose=True):
        if verbose:
            print("=" * 85)
            print(f"[ORCHESTRATOR]   → Received question: '{user_question}'")
            print(f"[ORCHESTRATOR]   → Routing to {self.generator.name} (Parametric Memory, No RAG)...")
        
        # Generate the response using parametric weights only
        raw_answer = self.generator.generate_response(user_question)
        if verbose:
            print(f"[{self.generator.name}] → Generated answer: {raw_answer}")
            print(f"[ORCHESTRATOR]   → Dispatching answer to {self.critic.name} for evaluation...")
        
        # Evaluate the answer with the Critic agent
        evaluation_result = self.critic.evaluate(user_question, raw_answer)
        if verbose:
            print(
                f"[{self.critic.name}]   → Faithfulness: {evaluation_result['faithfulness']:.2f} | "
                f"Completeness: {evaluation_result['completeness']:.2f} | "
                f"Hallucination Flag: {evaluation_result['hallucinated']}"
            )
            print("=" * 85)
            
        return {
            "question": user_question,
            "answer": raw_answer,
            "faithfulness": evaluation_result["faithfulness"],
            "completeness": evaluation_result["completeness"],
            "hallucinated": evaluation_result["hallucinated"]
        }

# Demonstrate a sample run of System 1 on Question 1
system1_orchestrator = OrchestratorSystem1()
sample_q1 = "What is the recommended dosage of Metformin for Type 2 diabetes?"
system1_sample_output = system1_orchestrator.process_query(sample_q1, verbose=True)
"""
add_code_cell(nb, sec6_code)

save_notebook(nb)
print("Section 6 appended successfully. Total cells:", len(nb["cells"]))
