import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class FinalEnsemble:
    def __init__(self):
        self.models = []
        self.model_names = []
        self.class_names = ['Burn Mark', 'Coating_defects', 'Crack', 'EROSION']
        
        self._load_models()
    
    def _load_models(self):
        model_configs = [
            {'path': 'turbine_model/augmented_training_v2/weights/best.pt', 'name': 'v2', 'weight': 1.0},
            {'path': 'turbine_model/augmented_training_v3/weights/best.pt', 'name': 'v3', 'weight': 1.0},
            {'path': 'turbine_model/augmented_training_yolo8n_v1/weights/best.pt', 'name': 'yolo8n', 'weight': 0.8},
        ]
        
        for config in model_configs:
            if os.path.exists(config['path']):
                try:
                    model = YOLO(config['path'])
                    model.model.cuda()  # На GPU
                    self.models.append({
                        'model': model,
                        'name': config['name'],
                        'weight': config['weight']
                    })
                    print(f"Загружена: {config['name']} ({config['path']})")
                except Exception as e:
                    print(f"Ошибка загрузки {config['name']}: {e}")
        
        print(f"Ensemble готов! Моделей: {len(self.models)}")
    
    def predict(self, image, conf_threshold=0.25):
        all_detections = []
        
        for model_info in self.models:
            try:
                results = model_info['model'](image, conf=conf_threshold, device=0, verbose=False)
                
                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        detection = {
                            'xyxy': box.xyxy.cpu().numpy()[0],
                            'conf': box.conf.cpu().numpy()[0] * model_info['weight'],  # Взвешенная уверенность
                            'cls': box.cls.cpu().numpy()[0],
                            'model': model_info['name']
                        }
                        all_detections.append(detection)
            except Exception as e:
                print(f"Ошибка в модели {model_info['name']}: {e}")
        
        # Применяем NMS к объединенным детекциям
        final_detections = self._apply_nms(all_detections)
        
        # Визуализация
        result_image = self._visualize_detections(image, final_detections)
        
        return result_image, final_detections
    
    def _apply_nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['conf'], reverse=True)
        final_detections = []
        
        while detections:
            current = detections[0]
            final_detections.append(current)
            
            detections = [det for det in detections[1:] 
                         if self._calculate_iou(current['xyxy'], det['xyxy']) < iou_threshold]
        
        return final_detections
    
    def _calculate_iou(self, box1, box2):
        """Вычисление IoU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _visualize_detections(self, image, detections):
        """Визуализация детекций"""
        result_image = image.copy()
        colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 0)]  # Зеленый, Желтый, Красный, Синий
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['xyxy'])
            cls_id = int(det['cls'])
            conf = det['conf']
            
            color = colors[cls_id]
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            
            # Рисуем bbox и подпись
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image

def test_final_ensemble():
    """Тестирование финального ensemble"""
    print("Запуск Final Ensemble...")
    ensemble = FinalEnsemble()
    
    if not ensemble.models:
        print("Не найдено моделей для ensemble!")
        return
    
    # Ищем тестовое изображение
    test_folders = ["dataset/test/images/", "dataset/valid/images/", "dataset/train/images/"]
    
    for folder in test_folders:
        if os.path.exists(folder):
            images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                test_path = os.path.join(folder, images[0])
                print(f"🔍 Тестируем на: {test_path}")
                
                image = cv2.imread(test_path)
                if image is not None:
                    result_img, detections = ensemble.predict(image)
                    
                    print(f"📊 Найдено дефектов: {len(detections)}")
                    for det in detections:
                        cls_name = ensemble.class_names[int(det['cls'])]
                        print(f"   - {cls_name}: {det['conf']:.3f} (модель: {det['model']})")
                    
                    # Сохраняем
                    os.makedirs("demo_results", exist_ok=True)
                    output_path = "demo_results/final_ensemble_result.jpg"
                    cv2.imwrite(output_path, result_img)
                    print(f"✅ Результат сохранен: {output_path}")
                    
                    # Показываем статистику
                    print(f"\n📈 Статистика ensemble:")
                    for i, name in enumerate(ensemble.class_names):
                        count = sum(1 for det in detections if int(det['cls']) == i)
                        print(f"   {name}: {count} детекций")
                    break

if __name__ == "__main__":
    test_final_ensemble()
