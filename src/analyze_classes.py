import os
from collections import Counter

def analyze_class_balance():
    """Анализируем баланс классов"""
    
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
    
    # Рекомендации
    print("\n🎯 Рекомендации:")
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)
    print(f"   Самый редкий: {class_names[min_class]} ({class_counts[min_class]} примеров)")
    print(f"   Самый частый: {class_names[max_class]} ({class_counts[max_class]} примеров)")

analyze_class_balance()