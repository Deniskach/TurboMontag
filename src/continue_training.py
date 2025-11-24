from ultralytics import YOLO
import torch

def train_100_epochs():
    print("🚀 ЗАПУСК ОБУЧЕНИЯ НА 100 ЭПОХ С ПРЕДОБУЧЕННЫМИ ВЕСАМИ")
    print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
    
    model = YOLO('turbine_model/gpu_training_v1/weights/best.pt')
    
    print("📊 Начинаем обучение с лучших весов...")
    
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,     
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        lr0=0.01,    
        patience=25,
        save=True,
        exist_ok=True,
        verbose=True,
        project='turbine_model',
        name='gpu_training_v2'
    )
    
    print("✅ Обучение на 100 эпох завершено!")

if __name__ == "__main__":
    train_100_epochs()