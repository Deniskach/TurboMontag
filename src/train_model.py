import os
import yaml
from ultralytics import YOLO
import torch
import gc

def train_model():
    # Очистка памяти перед началом
    torch.cuda.empty_cache()
    gc.collect()

    # Проверка GPU
    print(f"🔧 Используется: {torch.cuda.get_device_name(0)}")
    print(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Проверка датасета
    dataset_path = "dataset/data.yaml"
    if not os.path.exists(dataset_path):
        print("❌ Файл data.yaml не найден!")
        return
    
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"📊 Обнаружены классы: {data['names']}")
    print(f"🎯 Количество классов: {data['nc']}")
    
    # Создаем модель
    model = YOLO('yolov8n.pt')
    
    # Параметры обучения
    print("\n   Параметры обучения:")
    print("   - Модель: YOLOv8n")
    print("   - Эпохи: 100")
    print("   - Batch size: 8-16")
    print("   - Размер изображения: 640px")
    
    try:
        print("\n🎓 Начинаем обучение...")
        results = model.train(
            data=dataset_path,
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,           # Используем GPU 0
            workers=2,
            lr0=1e-3,
            patience=20,
            save=True,
            exist_ok=True,      # Перезаписывать существующие результаты
            verbose=True,
            project='turbine_model',
            name='augmented_training_yolo8n_v1',
            optimizer='AdamW',
            cache=False,
            amp=False
            #close_mosaic=5,
            #overlap_mask=False,
            #plots=True,
            # Аугментация для экономии памяти
            #hsv_h=0.01,
            #hsv_s=0.5,          
            #hsv_v=0.3,
            #degrees=5.0,
            #translate=0.05,
            #scale=0.3,
            #shear=2.0,
            #perspective=0.0005,
            #flipud=0.0,
            #mosaic=0.8,
            #mixup=0.0,
            #copy_paste=0.0
        )
        
        print("\n✅ Обучение завершено успешно!")
        print("📁 Модель сохранена в: turbine_model/augmented_training_new_data_yolo8n_v1/")
        
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        print("Пробуем уменьшить batch size...")
        
        # Пробуем с меньшим batch size
        try:
            results = model.train(
                data=dataset_path,
                epochs=100,
                imgsz=640,
                batch=8,        # Уменьшаем batch size
                device=0,
                workers=1,
                lr0=1e-3,
                patience=20,
                save=True,
                project='turbine_model',
                name='augmented_training_yolo8n_v1',
                optimizer='AdamW',
                cache=False,
                amp=False
                #close_mosaic=3,
                # Аугментация для экономии памяти
                #hsv_h=0.005,
                #hsv_s=0.3, 
                #hsv_v=0.2,
                #degrees=2.0,
                #translate=0.02,
                #scale=0.2,
                #shear=1.0,
                #perspective=0.0001,
                #flipud=0.0,
                #mosaic=0.5,     # ⚠️ Уменьшаем mosaic
                #mixup=0.0,
                #copy_paste=0.0
            )
        except Exception as e2:
            print(f"❌ Критическая ошибка: {e2}")

if __name__ == "__main__":
    train_model()