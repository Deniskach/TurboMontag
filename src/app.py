from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime
import logging
import os

# Импортируем модель
from ensemble import FinalEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ТАГАТ - Система контроля качества ГТД")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Инициализация модели
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    try:
        print("🔄 Загрузка Ensemble моделей...")
        analyzer = FinalEnsemble()
        print("✅ Ensemble модели успешно загружены!")
        print(f"📊 Загружено моделей: {len(analyzer.models)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        analyzer = None

class DefectAnalyzer:
    def __init__(self, ensemble_model):
        self.model = ensemble_model
        self.defect_mapping = {
            'Burn Mark': {'name': 'Прожог', 'criticality': 'Высокий'},
            'Coating_defects': {'name': 'Дефект покрытия', 'criticality': 'Средний'},
            'Crack': {'name': 'Трещина', 'criticality': 'Критический'},
            'EROSION': {'name': 'Эрозия', 'criticality': 'Высокий'}
        }
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Предобработка изображения для модели"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            if image_np.shape[-1] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            return image_np
        except Exception as e:
            logger.error(f"Ошибка предобработки изображения: {e}")
            raise
    
    def analyze_defects(self, image_np: np.ndarray) -> dict:
        """Анализ изображения на наличие дефектов"""
        try:
            if self.model is None:
                raise Exception("Модель не загружена")
            
            logger.info("🎯 Запуск предсказания модели...")
            result_img, detections = self.model.predict(image_np, conf_threshold=0.25)
            logger.info(f"📊 Найдено детекций: {len(detections)}")
            
            if detections:
                logger.info(f"Тип conf: {type(detections[0]['conf'])}")
                logger.info(f"Тип xyxy: {type(detections[0]['xyxy'])}")

            # Преобразуем детекции в нужный формат
            formatted_defects = []
            for i, det in enumerate(detections):
                cls_id = int(det['cls'])
                class_name_en = self.model.class_names[cls_id]
                
                defect_info = self.defect_mapping.get(class_name_en, {
                    'name': class_name_en, 
                    'criticality': 'Средний'
                })
                
                x1, y1, x2, y2 = map(float, det['xyxy'])
                
                defect_data = {
                    'id': i + 1,
                    'type': defect_info['name'],
                    'type_en': class_name_en,
                    'coordinates': {
                        'x': round(float((x1 + x2) / 2), 1),
                        'y': round(float((y1 + y2) / 2), 1)
                    },
                    'size': float(round(max(x2 - x1, y2 - y1), 1)),
                    'criticality': defect_info['criticality'],
                    'confidence': float(det['conf']),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'model_source': det.get('model', 'ensemble')
                }
                formatted_defects.append(defect_data)
                logger.info(f"   - {defect_info['name']}: {det['conf']:.3f}")
            
            # Сортируем дефекты по критичности
            criticality_order = {'Критический': 0, 'Высокий': 1, 'Средний': 2, 'Низкий': 3}
            formatted_defects.sort(key=lambda x: (
                criticality_order[x['criticality']], 
                -x['confidence']
            ))
            
            critical_defects = len([d for d in formatted_defects if d['criticality'] in ['Критический', 'Высокий']])
            
            return {
                'defects_found': len(formatted_defects),
                'critical_defects': critical_defects,
                'defects': formatted_defects,
                'analysis_id': f"ANL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'model_used': 'FinalEnsemble'
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа изображения: {e}")
            raise
    
    def draw_defects_on_image(self, image_np: np.ndarray, defects: list) -> bytes:
        """Отрисовка дефектов на изображении"""
        try:
            image_with_defects = image_np.copy()
            
            colors = {
                'Критический': (0, 0, 255),
                'Высокий': (0, 165, 255),
                'Средний': (0, 255, 255),
                'Низкий': (0, 255, 0)
            }
            
            for defect in defects:
                color = colors.get(defect['criticality'], (255, 255, 255))
                bbox = defect['bbox']
                
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    cv2.rectangle(
                        image_with_defects,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )
                    
                    label_en = f"{defect['type_en']} {defect['confidence']:.0%}"
                    cv2.putText(
                        image_with_defects,
                        label_en,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA
                    )
                else:
                    logger.warning(f"Некорректный bbox: {bbox}")
            
            _, buffer = cv2.imencode('.jpg', image_with_defects, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Ошибка отрисовки дефектов: {e}")
            raise

# Инициализация анализатора
defect_analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer, defect_analyzer
    try:
        print("🔄 Загрузка Ensemble моделей...")
        analyzer = FinalEnsemble()
        defect_analyzer = DefectAnalyzer(analyzer)
        print("✅ Ensemble модели успешно загружены!")
        print(f"📊 Загружено моделей: {len(analyzer.models)}")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        analyzer = None
        defect_analyzer = None

# API endpoints
@app.get("/")
async def root():
    return {"message": "ТАГАТ - Система контроля качества ГТД API", "status": "active"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_status": "loaded" if analyzer else "failed",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze-image")
async def analyze_image(
    engine_number: str = "ТАГАТ-2024-001",
    blade_number: str = "LP-001",
    file: UploadFile = File(...)
):
    """Анализ изображения на наличие дефектов"""
    try:
        if defect_analyzer is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")
        
        logger.info(f"🎯 Анализ для {engine_number}, лопатка {blade_number}")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")
        
        image_data = await file.read()
        image_np = defect_analyzer.preprocess_image(image_data)
        
        analysis_result = defect_analyzer.analyze_defects(image_np)
        annotated_image = defect_analyzer.draw_defects_on_image(image_np, analysis_result['defects'])
        
        analysis_result['annotated_image'] = f"data:image/jpeg;base64,{annotated_image}"
        analysis_result['engine_number'] = engine_number
        analysis_result['blade_number'] = blade_number
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

@app.post("/api/analyze-frame")
async def analyze_frame(
    engine_number: str = "ТАГАТ-2024-001",
    blade_number: str = "LP-001",
    image_data: str = None
):
    """Анализ кадра из видео"""
    try:
        if defect_analyzer is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")
            
        if not image_data:
            raise HTTPException(status_code=400, detail="Отсутствуют данные изображения")
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_np = defect_analyzer.preprocess_image(image_bytes)
        
        analysis_result = defect_analyzer.analyze_defects(image_np)
        annotated_image = defect_analyzer.draw_defects_on_image(image_np, analysis_result['defects'])
        
        analysis_result['annotated_image'] = f"data:image/jpeg;base64,{annotated_image}"
        analysis_result['engine_number'] = engine_number
        analysis_result['blade_number'] = blade_number
        
        return JSONResponse(content=analysis_result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа кадра: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа: {str(e)}")

if __name__ == "__main__":
    print("🚀 Запуск FastAPI сервера...")
    print("📝 Документация API: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")