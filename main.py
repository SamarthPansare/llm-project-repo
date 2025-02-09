from pdf2image import convert_from_path
import pytesseract
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import os
import json
from multiprocessing import Pool, cpu_count


# Load a local sentence-transformers model
model = SentenceTransformer('all-MiniLM-L6-v2')

dimension = 384  # Dimension of the embeddings

index_file = "faiss_index.index"

# Check if the index file exists
if os.path.exists(index_file):
    print(f"Loading index from {index_file}...")
    index = faiss.read_index(index_file)
else:
    print(f"Index file does not exist. Creating a new IndexFlatL2 with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)

# File to store the text_list
text_list_file = "text_list.json"


# Save text_list to a JSON file
def save_text_list():
    with open(text_list_file, "w") as f:
        json.dump(text_list, f)


# Load text_list from a JSON file
def load_text_list():
    if os.path.exists(text_list_file):
        try:
            with open(text_list_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {text_list_file} contains invalid JSON. Initializing with an empty list.")
            return []
    return []


# Load text_list at the start of the script
text_list = load_text_list()


# File to store processed PDFs
processed_files_file = "processed_files.json"

# Check if the processed_files_file exists, and create it if it doesn't
if not os.path.exists(processed_files_file):
    with open(processed_files_file, "w") as f:
        json.dump([], f)  # Initialize with an empty list

# Load processed files if the file exists
if os.path.exists(processed_files_file):
    with open(processed_files_file, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = []


def ocr_page(page_image):
    """Perform OCR on a single page image."""
    return pytesseract.image_to_string(page_image)


def pdf_to_text_parallel(pdf_path):
    """Convert PDF to text using multiprocessing for OCR."""
    try:
        # Convert PDF to images
        pages = convert_from_path(pdf_path, dpi=100)  # Reduce DPI for faster processing
        print(f"Extracting text from {len(pages)} pages...")

        # Use multiprocessing to perform OCR on each page
        with Pool(cpu_count()) as pool:
            text_chunks = pool.map(ocr_page, pages)

        # Combine text chunks from all pages
        text = " ".join(text_chunks)
        return text
    except FileNotFoundError:
        print(f"Error: File {pdf_path} not found.")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""


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
    embeddings = model.encode(text, batch_size=32, show_progress_bar=True)
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
        # Save the updated text_list to the JSON file
        save_text_list()
    else:
        print("No new chunks added to FAISS index.")
    print("-" * 30)


# query the faiss to get the index
def query_faiss(query, k=5):
    query_embedding = np.array(get_embeddings(query)).reshape(1, -1).astype('float32')
    print(f"query embedding shape {query_embedding.shape}")
    index = faiss.read_index("faiss_index.index")
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
        print("Prompt: ", prompt)
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
            # for chunk in chunks:
                # print("\n", chunk)
            print(f"\nSplitted text from file {filename} text into {len(chunks)} chunks")
            print("-" * 30)
            add_to_faiss_index(chunks)

        processed_files.append(filename)
        with open(processed_files_file, "w") as f:
            json.dump(processed_files, f)


# reading files from data dir and  calling process function
if __name__ == "__main__":
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
