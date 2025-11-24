from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Добавлены звездочки
    allow_credentials=True,
    allow_methods=["*"],  # Добавлены звездочки
    allow_headers=["*"],  # Добавлены звездочки
)

@app.get("/")  # Добавлен путь "/"
async def root():
    return {"message": 'Сервер работает!'}  # Добавлены кавычки для ключа

@app.post("/api/test-upload")  # Добавлен путь в кавычках
async def test_upload(file: UploadFile = File(...)):  # Добавлено двоеточие и закрывающая скобка
    return {
        "filename": file.filename,  # Добавлены кавычки и запятая
        "size": "ok"  # Добавлены кавычки
    }

if __name__ == "__main__":  # Добавлены кавычки
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Добавлены кавычки