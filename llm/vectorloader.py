import os
import time
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader
)

# === โหลด Excel แบบ custom ===
def load_excel_files(folder_path):
    print("📄 Loading Excel files...")
    start = time.time()
    docs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xlsx"):
                path = os.path.join(root, file)
                xls = pd.ExcelFile(path)
                for sheet_name in xls.sheet_names:
                    df = xls.parse(sheet_name)
                    text = df.to_string(index=False)
                    metadata = {"source": path, "sheet": sheet_name}
                    docs.append(Document(page_content=text, metadata=metadata))
    print(f"✅ Loaded {len(docs)} Excel documents in {time.time() - start:.2f} seconds\n")
    return docs

# === โหลดไฟล์เอกสารทั้งหมด ===
all_documents = []

loaders = [
    ("Text (.txt)", DirectoryLoader("documents", glob="**/*.txt", loader_cls=TextLoader)),
    ("PDF (.pdf)", DirectoryLoader("documents", glob="**/*.pdf", loader_cls=PyMuPDFLoader)),
    ("Word (.docx)", DirectoryLoader("documents", glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader)),
]

for name, loader in loaders:
    print(f"📄 Loading {name} files...")
    start = time.time()
    try:
        docs = loader.load()
        all_documents.extend(docs)
        print(f"✅ Loaded {len(docs)} {name} documents in {time.time() - start:.2f} seconds\n")
    except Exception as e:
        print(f"❌ Error loading {name} documents: {e}")

# เพิ่ม Excel
all_documents.extend(load_excel_files("documents"))

# === แบ่งเนื้อหา ===
print("🔪 Splitting documents...")
start = time.time()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(all_documents)
print(f"✅ Split into {len(chunks)} chunks in {time.time() - start:.2f} seconds\n")

# === เตรียม texts และ metadata ===
texts = []
metadatas = []
for chunk in chunks:
    content = f"passage: {chunk.page_content.strip()}"
    texts.append(content)
    metadatas.append(chunk.metadata)

# === โหลดและใช้ SentenceTransformer ด้วย CPU ===
start = time.time()
#model = SentenceTransformer("intfloat/multilingual-e5-large")
model = SentenceTransformer("BAAI/bge-m3", device="cpu")
embeddings = model.encode(texts, normalize_embeddings=True)
print(f"✅ Finished embedding {len(texts)} passages in {time.time() - start:.2f} seconds\n")

# === สร้างและบันทึก FAISS Index ===
print("💾 Saving FAISS index and metadata...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss.index")
with open("faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadatas, f)
with open("faiss_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Index and metadata saved successfully.")
