from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
from transformers import pipeline
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

uploaded_text = ""

def extract_text_from_pdf(pdf_bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global uploaded_text
    try:
        pdf_bytes = await file.read()
        uploaded_text = extract_text_from_pdf(pdf_bytes)
        logging.info("PDF uploaded and text extracted")
        return {"status": "success", "text": uploaded_text[:500]}
    except Exception as e:
        logging.error(f"Error uploading PDF: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/answer/")
async def get_answer(question: str = Form(...)):
    global uploaded_text
    if not uploaded_text:
        return {"status": "error", "message": "No document uploaded. Please upload a PDF first."}

    try:
        results = qa_pipeline({"question": question, "context": uploaded_text}, top_k=3)
        return {"answers": [{"text": r["answer"], "score": r["score"]} for r in results]}
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
