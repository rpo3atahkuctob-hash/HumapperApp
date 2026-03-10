// src/js/main.js
// Инициализация простого UI для выбора модели и загрузки через viewer.js

import { loadModel, getModels } from './viewer.js';

async function init() {
  const models = await getModels();
  const select = document.getElementById('model-select');
  if (!select) return;

  // Заполнить селект
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.path;
    opt.textContent = m.title || m.id;
    select.appendChild(opt);
  });

  // Загрузка первой модели по умолчанию
  if (models.length) loadModel(models[0].path);

  select.addEventListener('change', (e) => {
    loadModel(e.target.value);
  });
}

// Автоинициализация при загрузке страницы
window.addEventListener('DOMContentLoaded', init);

export { init };

