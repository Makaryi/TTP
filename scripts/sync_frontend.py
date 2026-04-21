#!/usr/bin/env python3
"""Синхронизирует templates/ -> frontend/ перед деплоем на Cloudflare Pages."""
import os, shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_TPL = os.path.join(BASE, 'web_app', 'templates')
SRC_STC = os.path.join(BASE, 'web_app', 'static')
DST     = os.path.join(BASE, 'frontend')

os.makedirs(DST, exist_ok=True)

for fname in os.listdir(SRC_TPL):
    if fname.endswith('.html') and not fname.startswith('index_improved'):
        dst_name = fname.replace('lie_detector', 'lie-detector')
        shutil.copy2(os.path.join(SRC_TPL, fname), os.path.join(DST, dst_name))

dst_static = os.path.join(DST, 'static')
if os.path.exists(dst_static):
    shutil.rmtree(dst_static)
shutil.copytree(SRC_STC, dst_static)

print('✅ frontend/ синхронизирован')
