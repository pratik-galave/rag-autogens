from manage_notebook import load_or_create_notebook, save_notebook, add_markdown_cell, add_code_cell

nb = load_or_create_notebook()

# Section 4: Build the Embedding Model and Vector Database
sec4_md = """# Section 4: Build the Embedding Model and Vector Database

In this section, we build the core semantic retrieval engine:
1. **Sentence-Transformers Embedding Model:** We load `all-MiniLM-L6-v2`. This model maps text sentences and paragraphs into a 384-dimensional dense vector space where semantically similar concepts reside close to one another.
2. **Dense Vector Representations:** We convert all medical chunks in our knowledge base into 384-dimensional floating-point vectors.
3. **FAISS Vector Index:** We initialize an in-memory `IndexFlatL2` index using Facebook AI Similarity Search (FAISS). This performs exact Euclidean distance ($L_2$) nearest-neighbor search with microsecond latency on CPU.
4. **Retrieval Function:** We construct a dedicated query function that takes any natural language query, generates its embedding, searches the FAISS index for the top-$k$ nearest neighbors, and returns the grounded text chunks.
"""
add_markdown_cell(nb, sec4_md)

sec4_code = """# Define the name of the lightweight, high-performance open-source embedding model
embedding_model_name = "all-MiniLM-L6-v2"

# Load the SentenceTransformer model onto the local CPU
embedding_model = SentenceTransformer(embedding_model_name)

# Convert all text passages in our medical knowledge base into dense vector embeddings
knowledge_embeddings = embedding_model.encode(medical_knowledge_base, show_progress_bar=False)

# Convert the resulting numpy embeddings array to float32 (the required precision for FAISS)
knowledge_embeddings = np.array(knowledge_embeddings).astype("float32")

# Extract the dimensionality of the vector space (384 dimensions for all-MiniLM-L6-v2)
vector_dimension = knowledge_embeddings.shape[1]

# Initialize a FAISS Flat L2 distance index for exact vector similarity search
faiss_index = faiss.IndexFlatL2(vector_dimension)

# Add all passage vector embeddings into the FAISS index
faiss_index.add(knowledge_embeddings)

# Print confirmation of the vector database build
print(f" FAISS Index built successfully!")
print(f" Total indexed vectors: {faiss_index.ntotal}")
print(f" Embedding dimensions per vector: {vector_dimension}")

# Define a modular retrieval function to fetch the top-k most relevant text chunks for any query
def retrieve_relevant_chunks(user_query, top_k=2):
    # Convert the incoming user question into an embedding vector
    query_vector = embedding_model.encode([user_query])
    
    # Cast the query vector to float32 as required by FAISS
    query_vector = np.array(query_vector).astype("float32")
    
    # Query the FAISS index to find the top-k nearest neighbor vector distances and indices
    distances, indices = faiss_index.search(query_vector, top_k)
    
    # Gather the matching text chunks from our knowledge base using the retrieved indices
    retrieved_chunks = [medical_knowledge_base[idx] for idx in indices[0] if idx < len(medical_knowledge_base)]
    
    # Return both the text chunks and their corresponding L2 distances
    return retrieved_chunks, distances[0]

# Perform a quick verification test of the retrieval engine with a sample query
test_query = "What is the starting dose of Metformin?"
matched_chunks, match_scores = retrieve_relevant_chunks(test_query, top_k=1)

# Display the top retrieved result to verify correctness
print("-" * 75)
print(f"Verification Test Query: '{test_query}'")
print(f"FAISS Nearest Neighbor Distance: {match_scores[0]:.4f}")
print("Retrieved Chunk Preview:")
print(matched_chunks[0][:180] + "...")
print("-" * 75)
"""
add_code_cell(nb, sec4_code)

save_notebook(nb)
print("Section 4 appended successfully. Total cells:", len(nb["cells"]))
