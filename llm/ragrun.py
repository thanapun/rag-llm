import pickle
import faiss
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

# โหลด FAISS Index
index = faiss.read_index("faiss.index")

# โหลด texts และ metadata แยก
with open("faiss_texts.pkl", "rb") as f:
    texts = pickle.load(f)

with open("faiss_metadata.pkl", "rb") as f:
    metadatas = pickle.load(f)

# โหลด embedding model (e5 รองรับไทย+แม่นยำ)
embed_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")
# embed_model = SentenceTransformer("BAAI/bge-m3")

# โหลด LLM
# llm = OllamaLLM(model="openchat")  # ธรรมชาติแต่มั่วเยอะ
# llm = OllamaLLM(model="nous-hermes2")  # ภาษาไทยไม่ดี
llm = OllamaLLM(model="llama3")  # คุณภาพดี เหมาะ production


# llm = OllamaLLM(model="mixtral") #ใช้ GPU 24GB

# ฟังก์ชันถาม-ตอบแบบอ้างอิงจากเอกสารเท่านั้น
def ask_rag(query):
    # เตรียม embedding ด้วย prefix ตาม e5
    query_vec = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
    D, I = index.search(query_vec, k=10)

    # รวม context และอ้างอิงแหล่งที่มา
    retrieved_docs = []
    for i in I[0]:
        content = texts[i].replace("passage: ", "").strip()
        metadata = metadatas[i]
        source = metadata.get("source", "ไม่ระบุ")
        sheet = metadata.get("sheet", "ไม่ระบุ")
        retrieved_docs.append(f"[จากไฟล์: {source}, sheet: {sheet}]\n{content}")

    context = "\n\n".join(retrieved_docs)

    # Prompt ที่เน้นความแม่นยำและอ้างอิงเท่านั้น
    prompt = f"""
        คุณคือผู้ช่วยที่เชี่ยวชาญการค้นหาข้อมูลจากเอกสารเท่านั้น ห้ามแต่งเนื้อหา หรือคาดเดาคำตอบ

        -------------------------------
        เอกสารอ้างอิง:
        {context}
        -------------------------------

        คำถาม:
        {query}

        กรุณาตอบเป็นภาษาไทยโดยละเอียด และอ้างอิงจากแหล่งข้อมูลด้านบนอย่างชัดเจน (ระบุชื่อไฟล์และ sheet ทุกครั้ง)
            """
    return llm.invoke(prompt)


# วนลูปรับคำถามจากผู้ใช้
if __name__ == "__main__":
    while True:
        q = input("ถาม: ")
        if q.lower().strip() == "exit":
            break
        print("ตอบ:", ask_rag(q))
