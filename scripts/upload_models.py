#!/usr/bin/env python3
"""
upload_models.py — Загрузить ML-модели на HuggingFace Hub.

Запуск:
    pip install huggingface_hub
    huggingface-cli login
    python scripts/upload_models.py --repo YOUR_USERNAME/emotion-models

Документация: https://huggingface.co/docs/huggingface_hub/guides/upload
"""
import os
import argparse
from pathlib import Path


def upload_models(repo_id: str, models_dir: str = "models", private: bool = True):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    print(f"\n{'='*55}")
    print(f"  HuggingFace Upload")
    print(f"  Repo: {repo_id}")
    print(f"  Dir:  {models_dir}")
    print(f"{'='*55}\n")

    # Создаём репозиторий если не существует
    try:
        create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)
        print(f"✅ Репозиторий готов: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️  create_repo: {e}")

    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"❌ Папка {models_dir} не найдена")
        return

    # Загружаем все .h5 файлы
    uploaded = 0
    for fname in sorted(models_path.glob("*.h5")):
        size_mb = fname.stat().st_size / 1024 / 1024
        print(f"  📤 {fname.name} ({size_mb:.1f} MB)...", end=" ", flush=True)
        try:
            api.upload_file(
                path_or_fileobj=str(fname),
                path_in_repo=fname.name,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {fname.name}"
            )
            print("✅")
            uploaded += 1
        except Exception as e:
            print(f"❌ {e}")

    # Создаём README для HuggingFace
    readme = f"""---
license: mit
tags:
  - emotion-recognition
  - lie-detection
  - computer-vision
  - tensorflow
language:
  - ru
---

# AI Emotion & Lie Detector Models

Модели для определения эмоций по лицу и детектора лжи.

## Состав
- `best_emotion_model.h5` — CNN для распознавания 7 эмоций (FER2013)
- `lie_detector_model.h5` — Мультимодальная модель (видео + аудио)

## Использование
```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="{repo_id}", filename="best_emotion_model.h5")
```
"""
    try:
        api.upload_file(
            path_or_fileobj=readme.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add README"
        )
        print("  📝 README.md загружен")
    except Exception as e:
        print(f"  ⚠️  README: {e}")

    print(f"\n{'='*55}")
    print(f"  ✅ Загружено файлов: {uploaded}")
    print(f"  🔗 https://huggingface.co/{repo_id}")
    print(f"{'='*55}\n")
    print("Теперь задайте переменную окружения на Render:")
    print(f"  HF_REPO_ID = {repo_id}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Загрузить модели на HuggingFace Hub"
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g. username/emotion-models)"
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Папка с моделями (по умолчанию: models)"
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Сделать репозиторий публичным"
    )
    args = parser.parse_args()
    upload_models(args.repo, args.models_dir, private=not args.public)
