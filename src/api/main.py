from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Legal Document RAG Tool"}

@app.post("/analyze")
def analyze(query: str):
    return {"message": "Analyzing the document..."}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    return {"message": "Uploading the document..."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)