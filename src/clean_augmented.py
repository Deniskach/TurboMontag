import os

def clean_augmented_files():
    """Удаляем все аугментированные файлы"""
    
    for split in ['train', 'valid']:
        images_dir = f'dataset/{split}/images'
        labels_dir = f'dataset/{split}/labels'
        
        # Удаляем аугментированные изображения
        for file in os.listdir(images_dir):
            if file.startswith('aug_'):
                os.remove(os.path.join(images_dir, file))
                print(f"🗑️ Удален: {file}")
        
        # Удаляем аугментированные разметки
        for file in os.listdir(labels_dir):
            if file.startswith('aug_'):
                os.remove(os.path.join(labels_dir, file))
                print(f"🗑️ Удален: {file}")
    
    print("✅ Все аугментированные файлы удалены!")

def check_dataset_size():
    """Проверяем размер датасета после очистки"""
    from collections import Counter
    
    class_names = ['Burn Mark', 'Coating_defects', 'Crack', 'EROSION']
    class_counts = Counter()
    
    for split in ['train', 'valid']:
        labels_dir = f'dataset/{split}/labels'
        
        for label_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
    
    print("\n📊 Размер датасета после очистки:")
    total = sum(class_counts.values())
    for class_id, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"   {class_names[class_id]}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    clean_augmented_files()
    check_dataset_size()