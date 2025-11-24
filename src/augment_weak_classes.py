import os
import random
import shutil
from PIL import Image, ImageEnhance
from collections import Counter

def analyze_classes():
    """Анализ распределения классов"""
    class_names = ['Burn Mark', 'Coating_defects', 'Crack', 'EROSION']
    class_counts = Counter()
    
    for split in ['train', 'valid']:
        labels_dir = f'dataset/{split}/labels'
        
        for label_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
    
    print("📊 Распределение классов:")
    total = sum(class_counts.values())
    for class_id, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"   {class_names[class_id]}: {count} ({percentage:.1f}%)")
    
    return class_counts

def gentle_augment_weak_classes():
    """Мягкая аугментация только для слабых классов"""
    
    weak_classes_ids = [1, 3]  # Coating_defects и EROSION
    augmented_count = 0
    
    for split in ['train']:  # Только тренировочные данные
        images_dir = f'dataset/{split}/images'
        labels_dir = f'dataset/{split}/labels'
        
        # Только оригинальные файлы (не аугментированные)
        image_files = [f for f in os.listdir(images_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg')) 
                      and not f.startswith(('aug_', 'gentle_'))]
        
        print(f"🔍 Обработка {len(image_files)} изображений...")
        
        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            label_path = os.path.join(labels_dir, image_file.rsplit('.', 1)[0] + '.txt')
            
            if not os.path.exists(label_path):
                continue
                
            # Проверяем есть ли слабые классы
            has_weak_class = False
            with open(label_path, 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    if class_id in weak_classes_ids:
                        has_weak_class = True
                        break
            
            if not has_weak_class:
                continue
            
            # Загружаем и аугментируем
            try:
                original_image = Image.open(image_path)
                
                # Создаем только 1 аугментированную версию (вместо 3)
                augmented_image = original_image.copy()
                
                # ТОЛЬКО мягкие преобразования:
                transformations = []
                
                # 1. Яркость (50% chance) - небольшой диапазон
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Brightness(augmented_image)
                    factor = random.uniform(0.9, 1.1)  # Всего ±10%
                    augmented_image = enhancer.enhance(factor)
                    transformations.append(f"bright_{factor:.1f}")
                
                # 2. Контраст (50% chance) - небольшой диапазон
                if random.random() > 0.5:
                    enhancer = ImageEnhance.Contrast(augmented_image)
                    factor = random.uniform(0.9, 1.2)  # Всего +20%
                    augmented_image = enhancer.enhance(factor)
                    transformations.append(f"contrast_{factor:.1f}")
                
                # НЕТ поворотов, отражений, шума!
                
                # Сохраняем
                base_name = image_file.rsplit('.', 1)[0]
                extension = image_file.rsplit('.', 1)[1]
                new_image_name = f"gentle_aug_{base_name}.{extension}"
                new_image_path = os.path.join(images_dir, new_image_name)
                
                augmented_image.save(new_image_path, quality=95)
                
                # Копируем разметку
                new_label_path = os.path.join(labels_dir, f"gentle_aug_{base_name}.txt")
                shutil.copy2(label_path, new_label_path)
                
                augmented_count += 1
                transform_str = "+".join(transformations) if transformations else "original"
                print(f"✅ {new_image_name} [{transform_str}]")
                
            except Exception as e:
                print(f"❌ Ошибка с {image_file}: {e}")
    
    return augmented_count

if __name__ == "__main__":
    print("🔍 Анализ классов перед мягкой аугментацией...")
    initial_counts = analyze_classes()
    
    print("\n🎯 Запуск МЯГКОЙ аугментации...")
    total_augmented = gentle_augment_weak_classes()
    
    print(f"\n📈 Создано {total_augmented} мягко аугментированных примеров")
    
    print("\n🔍 Анализ классов после аугментации...")
    final_counts = analyze_classes()
    
    # Статистика
    print(f"\n📊 Улучшение для слабых классов:")
    class_names = ['Burn Mark', 'Coating_defects', 'Crack', 'EROSION']
    for class_id in [1, 3]:
        initial = initial_counts[class_id]
        final = final_counts[class_id]
        improvement = final - initial
        print(f"   {class_names[class_id]}: {initial} → {final} (+{improvement})")