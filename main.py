from pdf2image import convert_from_path
import pytesseract
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import os


# Load a local sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')
text_list = []

dimension = 384  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)


# convert pdf to text using OCR
def pdf_to_text(pdf_path):
    text = ""
    try:
        for page in convert_from_path(pdf_path, dpi=150):
            text += pytesseract.image_to_string(page)
    except FileNotFoundError:
        print(f"Error: File {pdf_path} not found.")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return text


# split the text into chunks with overlap for better context
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


# generate embeddings
def get_embeddings(text):
    # Generate embeddings using the local model
    embeddings = model.encode(text)
    return embeddings


# add chunks and their embedding to faiss
def add_to_faiss_index(chunks):
    new_entries = 0  # Count new entries added to the index
    for chunk in chunks:
        if chunk in text_list:  # Check if the chunk already exists
            continue

        embedding = get_embeddings(chunk)
        text_list.append(chunk)
        index.add(np.array([embedding], dtype=np.float32))
        new_entries += 1

    if new_entries > 0:
        faiss.write_index(index, "faiss_index.index")
        print(f"FAISS index updated and saved to disk. Size: {index.ntotal}")
    else:
        print("No new chunks added to FAISS index.")
    print("-" * 30)


# query the faiss to get the index
def query_faiss(query, k=5):
    query_embedding = np.array(get_embeddings(query)).reshape(1, -1).astype('float32')
    print(f"query embedding shape {query_embedding.shape}")
    distances, indices = index.search(query_embedding, k)
    print("Distances: ", distances)
    print("Indices: ", indices)
    results = [text_list[idx] for idx in indices[0] if idx != -1]
    return results, distances


# query deepseek model to get refined results
def query_deepseek(context, query):
    prompt = f"Context:\n{context} \nQuery:\n{query}"
    try:
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:1.5b"],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"DeepSeek error: {result.stderr.strip()}")
        return result.stdout.strip()
    except Exception as e:
        print(f"Error querying DeepSeek: {e}")
        return None


# process the pdfs from the data directory
def process_pdfs(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            print(f"Processing {pdf_path} ...")
            extracted_text = pdf_to_text(pdf_path)
            print(f"Extracted text from {filename}")
            chunks = chunk_text(extracted_text)
            print(f"Chunks from file {filename}:\n")
            for chunk in chunks:
                print("\n", chunk)
            print(f"\nSplitted text from file {filename} text into {len(chunks)} chunks")
            print("-" * 30)
            add_to_faiss_index(chunks)


# reading files from data dir and  calling process function
data_dir = "data"
print("-" * 30)
if not os.path.exists(data_dir):
    print(f"Directory {data_dir} does not exists")
else:
    process_pdfs(data_dir)


while True:
    query = input("\nEnter your Query: ")
    results, distances = query_faiss(query)
    print("-" * 30)
    print("Results: ", results)
    print("distances: ", distances)
    print(f"Top {len(results)} results retrieved from FAISS:\n")
    for i, (res, dist) in enumerate(zip(results, distances[0])):
        print(f"{i+1}] {res} (distance: {dist})")
    print("-" * 30)

    context = " ".join(results)
    response = query_deepseek(context, query)
    print("\n\n\nResponse From Deepseek:\n")
    print(response)
    print("-" * 30)
