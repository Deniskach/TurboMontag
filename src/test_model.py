from ultralytics import YOLO
import os
import random

def test_final_model():
    # Загружаем лучшую модель из 77 эпох
    model = YOLO('turbine_model/gpu_training_v2/weights/best.pt')
    
    print("🎯 ТЕСТИРУЕМ ФИНАЛЬНУЮ МОДЕЛЬ (77 ЭПОХ)")
    print("=" * 50)
    
    # Тестируем на нескольких случайных изображениях
    test_images = os.listdir('dataset/test/images')
    
    for i in range(3):  # Протестируем на 3 изображениях
        if test_images:
            test_image = random.choice(test_images)
            print(f"\n📸 Тест {i+1}: {test_image}")
            
            results = model.predict(
                source=f'dataset/test/images/{test_image}',
                save=True,
                conf=0.3,
                project='final_test_results',
                name=f'test_{i+1}'
            )
            
            for r in results:
                print(f"   Обнаружено объектов: {len(r.boxes)}")
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"   ▸ {class_name}: {confidence:.2f}")

if __name__ == "__main__":
    test_final_model()