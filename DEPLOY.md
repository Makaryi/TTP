# DEPLOY.md — Публикация AI Emotion & Lie Detector

Полностью бесплатная архитектура для публичного сайта:

```
GitHub (private repo)
       │
       ├── Backend API ──► Render.com        (бесплатно)
       │                        ↑
       ├── ML Models   ──► HuggingFace Hub   (бесплатно)
       │
       └── Frontend    ──► Cloudflare Pages  (бесплатно)
```

---

## Шаг 1 — Загрузить модели на HuggingFace

### 1.1 Создать аккаунт
Зарегистрируйся на https://huggingface.co

### 1.2 Авторизоваться локально
```bash
pip install huggingface_hub
huggingface-cli login
# вставь токен из https://huggingface.co/settings/tokens
```

### 1.3 Загрузить модели
```bash
python scripts/upload_models.py --repo ВАШ_ЮЗЕРНЕЙМ/emotion-models
```

Модели (~140 MB) будут доступны на:
`https://huggingface.co/ВАШ_ЮЗЕРНЕЙМ/emotion-models`

---

## Шаг 2 — Деплой Backend на Render.com

### 2.1 Создать аккаунт
Зарегистрируйся на https://render.com (через GitHub)

### 2.2 Новый Web Service
1. Dashboard → **New** → **Web Service**
2. Подключи GitHub репозиторий
3. Настройки:
   - **Root Directory:** `web_app`
   - **Build Command:** `pip install --upgrade pip && pip install -r ../requirements.prod.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Plan:** Free

### 2.3 Переменные окружения
В разделе **Environment** добавь:
```
HF_REPO_ID = ВАШ_ЮЗЕРНЕЙМ/emotion-models
```

После деплоя сервис будет доступен по адресу:
`https://emotion-ai-api.onrender.com`

Проверь: `https://emotion-ai-api.onrender.com/api/status`

---

## Шаг 3 — Деплой Frontend на Cloudflare Pages

### 3.1 Создать аккаунт
Зарегистрируйся на https://pages.cloudflare.com (через GitHub)

### 3.2 Обновить имя API-сервера
Открой `frontend/_redirects` и замени:
```
emotion-ai-api.onrender.com  →  ВАШ-АПП.onrender.com
```

### 3.3 Создать проект
1. Cloudflare Dashboard → **Workers & Pages** → **Create** → **Pages**
2. Подключи GitHub репозиторий
3. Настройки сборки:
   - **Build output directory:** `frontend`
   - **Build command:** `python scripts/sync_frontend.py`
   - **Root directory:** `/` (корень репо)

### 3.4 Готово!
Сайт будет доступен по адресу:
`https://emotion-ai.pages.dev`

---

## Шаг 4 — GitHub Secrets для CI/CD (опционально)

Для автоматического деплоя через GitHub Actions добавь в:
**GitHub → Settings → Secrets → Actions**

| Secret                  | Где взять                                        |
|------------------------|--------------------------------------------------|
| `CLOUDFLARE_API_TOKEN` | Cloudflare → My Profile → API Tokens            |
| `CLOUDFLARE_ACCOUNT_ID`| Cloudflare → правый sidebar → Account ID        |
| `RENDER_DEPLOY_HOOK`   | Render → Settings → Deploy Hook URL             |

---

## Структура репо

```
├── .github/workflows/ci.yml   # Авто-деплой
├── .gitignore                  # Исключает models/ и dataset/
├── Procfile                    # Для Render
├── render.yaml                 # Конфиг Render
├── requirements.prod.txt       # Prod зависимости
├── frontend/                   # Cloudflare Pages (статика)
│   ├── _redirects              # API прокси
│   ├── _headers                # Security headers
│   ├── index.html
│   └── static/
├── web_app/
│   ├── app.py                  # Production Flask app
│   ├── templates/              # HTML шаблоны
│   └── static/                 # CSS / JS
├── models/                     # НЕ в git (хранятся на HuggingFace)
└── scripts/
    ├── upload_models.py        # Загрузка моделей на HuggingFace
    └── sync_frontend.py        # Синхронизация templates → frontend/
```

---

## FAQ

**Q: Почему сервер на Render засыпает?**
A: Free tier засыпает через 15 мин без запросов. Первый запрос займёт ~30 сек.
Для демо — нормально. Для прода — платный план ($7/мес).

**Q: Модели не скачиваются при старте?**
A: Убедись что `HF_REPO_ID` задан в Render Environment Variables.

**Q: CORS ошибка в браузере?**
A: `flask-cors` уже добавлен в `app.py`. Убедись что `_redirects` 
в Cloudflare Pages корректно указывает на твой Render URL.
