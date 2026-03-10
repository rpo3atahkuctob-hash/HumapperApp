// src/js/viewer.js
// Небольшая обёртка для загрузки .glb моделей в <model-viewer> или для интеграции с three.js

async function getModels() {
  try {
    const res = await fetch('/src/models/models.json');
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    console.warn('Не удалось загрузить models.json', e);
    return [];
  }
}

function loadModel(path) {
  // Предпочтительный вариант: <model-viewer id="model"> в index.html
  const mv = document.getElementById('model-viewer') || document.getElementById('model');
  if (mv) {
    // для <model-viewer> используем атрибут src
    if ('setAttribute' in mv) mv.setAttribute('src', path);
    else mv.src = path;
    return;
  }

  // Если вы используете three.js, сюда можно подключить GLTFLoader
  console.warn('Нет <model-viewer> с id="model-viewer" или id="model" на странице. Реализуйте загрузку для three.js.');
}

export { getModels, loadModel };

