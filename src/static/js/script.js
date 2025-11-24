// Загрузка информации о моделях
async function loadModelInfo() {
  try {
    const response = await fetch('/api/models');
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    const data = await response.json();

    let html = `<p><strong>Моделей в ensemble:</strong> ${data.ensemble_size}</p>`;
    html += `<p><strong>Классы дефектов:</strong> ${data.classes.join(
      ', '
    )}</p>`;
    html += `<div class="mt-3"><strong>Модели:</strong><ul class="mt-2">`;

    data.models.forEach((model) => {
      const deviceIcon = model.device === 'cuda' ? '🎯' : '💻';
      html += `<li>${model.name} (вес: ${model.weight}, устройство: ${deviceIcon} ${model.device})</li>`;
    });

    html += `</ul></div>`;

    document.getElementById('modelInfo').innerHTML = html;
  } catch (error) {
    console.error('Error loading model info:', error);
    document.getElementById('modelInfo').innerHTML =
      '<div class="alert alert-warning">Не удалось загрузить информацию о моделях</div>';
  }
}

// Обработка формы загрузки изображения
document
  .getElementById('uploadForm')
  .addEventListener('submit', async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById('imageUpload');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const spinner = document.getElementById('spinner');
    const resultSection = document.getElementById('resultSection');
    const resultImage = document.getElementById('resultImage');
    const resultStats = document.getElementById('resultStats');
    const detectionsList = document.getElementById('detectionsList');

    if (!fileInput.files[0]) {
      showAlert('Пожалуйста, выберите изображение', 'warning');
      return;
    }

    // Валидация размера файла (макс 10MB)
    if (fileInput.files[0].size > 10 * 1024 * 1024) {
      showAlert('Размер файла не должен превышать 10MB', 'warning');
      return;
    }

    // Показываем загрузку
    analyzeBtn.disabled = true;
    spinner.style.display = 'inline-block';

    try {
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `Ошибка сервера: ${response.status}`
        );
      }

      const result = await response.json();

      // Показываем результат
      displayResults(result, resultImage, resultStats, detectionsList);
      resultSection.style.display = 'block';

      // Прокрутка к результатам
      resultSection.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
      console.error('Error:', error);
      showAlert(error.message, 'danger', resultStats);
      resultSection.style.display = 'block';
    } finally {
      analyzeBtn.disabled = false;
      spinner.style.display = 'none';
    }
  });

// Отображение результатов
function displayResults(result, resultImage, resultStats, detectionsList) {
  // Устанавливаем изображение с cache busting
  resultImage.src = result.urls.result + '?t=' + new Date().getTime();

  // Статистика
  resultStats.innerHTML = `
        <div class="alert alert-success">
            <h6><strong>✅ Анализ завершен!</strong></h6>
            <div class="row mt-2">
                <div class="col-6">
                    <small>Время обработки: <strong>${result.processing_time.toFixed(
                      2
                    )} сек</strong></small>
                </div>
                <div class="col-6">
                    <small>Найдено дефектов: <strong>${
                      result.detections_count
                    }</strong></small>
                </div>
            </div>
            <div class="row mt-1">
                <div class="col-6">
                    <small>Размер: ${result.image_size.width}×${
    result.image_size.height
  }</small>
                </div>
                <div class="col-6">
                    <small>ID запроса: ${result.request_id}</small>
                </div>
            </div>
        </div>
    `;

  // Статистика по классам
  let statsHtml = '<div class="row text-center mb-3">';
  for (const [className, count] of Object.entries(result.class_statistics)) {
    const badgeClass = count > 0 ? 'bg-success' : 'bg-secondary';
    statsHtml += `
            <div class="col">
                <span class="badge ${badgeClass} p-2">${className}: ${count}</span>
            </div>
        `;
  }
  statsHtml += '</div>';
  resultStats.innerHTML += statsHtml;

  // Список детекций
  if (result.detections.length > 0) {
    let detectionsHtml = '<h6>📋 Обнаруженные дефекты:</h6>';

    result.detections.forEach((det, index) => {
      const colors = ['success', 'warning', 'danger', 'info'];
      const icons = ['🔥', '🎨', '⚡', '💧']; // Иконки для классов
      const confidencePercent = (det.confidence * 100).toFixed(1);

      detectionsHtml += `
                <div class="detection-item alert alert-${colors[det.class_id]}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <strong>${icons[det.class_id]} ${
        det.class
      }</strong><br>
                            <small>Уверенность: <strong>${confidencePercent}%</strong></small><br>
                            <small>Модель: ${det.model}</small>
                        </div>
                        <span class="badge bg-dark">#${index + 1}</span>
                    </div>
                    <div class="mt-2">
                        <small class="text-muted">
                            Координаты: [${det.bbox.x1.toFixed(
                              0
                            )}, ${det.bbox.y1.toFixed(
        0
      )}] → [${det.bbox.x2.toFixed(0)}, ${det.bbox.y2.toFixed(0)}]
                        </small>
                    </div>
                </div>
            `;
    });

    detectionsList.innerHTML = detectionsHtml;
  } else {
    detectionsList.innerHTML = `
            <div class="alert alert-info text-center">
                <h6>🎉 Отличная новость!</h6>
                <p class="mb-0">Дефекты не обнаружены. Турбинная лопатка в отличном состоянии!</p>
            </div>
        `;
  }
}

// Вспомогательная функция для показа уведомлений
function showAlert(message, type = 'info', container = null) {
  const alertHtml = `
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

  if (container) {
    container.innerHTML = alertHtml;
  } else {
    // Создаем временное уведомление вверху страницы
    const alertDiv = document.createElement('div');
    alertDiv.innerHTML = alertHtml;
    alertDiv.style.position = 'fixed';
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.style.minWidth = '300px';
    document.body.appendChild(alertDiv);

    // Автоматически удаляем через 5 секунд
    setTimeout(() => {
      alertDiv.remove();
    }, 5000);
  }
}

// Предварительный просмотр изображения (опционально)
document.getElementById('imageUpload').addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      // Можно добавить предварительный просмотр если нужно
      console.log('Изображение выбрано:', file.name);
    };
    reader.readAsDataURL(file);
  }
});

// Загружаем информацию о моделях при старте
document.addEventListener('DOMContentLoaded', function () {
  loadModelInfo();

  // Показываем подсказку при первом посещении
  const isFirstVisit = !localStorage.getItem('visited');
  if (isFirstVisit) {
    setTimeout(() => {
      showAlert(
        '👋 Добро пожаловать! Загрузите изображение турбинной лопатки для анализа дефектов.',
        'info'
      );
      localStorage.setItem('visited', 'true');
    }, 1000);
  }
});
