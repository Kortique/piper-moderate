# Как обновлять ноды в Piper пайплайне

## API метод

```
PATCH /api/projects/{project_id}/patch/{revision}
Headers: User-Token: <TOKEN>, Content-Type: application/json
Body: jsondiffpatch delta (JSON)
```

**jsondiffpatch delta**: замена поля = `[old_value, new_value]`, добавление = `[new_value]`

## Шаги

1. **Получить текущую ревизию:**
```python
import httpx, json, os
from dotenv import load_dotenv
load_dotenv('/path/to/.env')
TOKEN = os.getenv('PIPER_TOKEN')
hdr = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
r = httpx.get('https://piper-next.artworks.ai/api/projects/d2911d10bb', headers=hdr, timeout=15)
live = r.json()
revision = live['revision']
pipeline = json.loads(live['pipeline']) if isinstance(live['pipeline'], str) else live['pipeline']
```

2. **Сформировать delta** (только изменённые поля):
```python
delta = {
    'pipeline': {
        'nodes': {
            'lgbm_evaluate': {
                'script': [old_script, new_script],           # замена
                'version': [old_version, new_version],        # замена
            },
            'ask_siglip2': {
                'inputs': {
                    'labels': {
                        'default': [old_labels_str, new_labels_str]
                    }
                }
            }
        },
        'flows': {
            'new_flow_key': [{'from': 'node_a', 'output': 'out', 'to': 'node_b', 'input': 'inp'}]  # добавление
        }
    }
}
```

3. **Отправить PATCH:**
```python
resp = httpx.patch(
    f'https://piper-next.artworks.ai/api/projects/d2911d10bb/patch/{revision}',
    headers=hdr,
    content=json.dumps(delta, ensure_ascii=False),
    timeout=30,
)
print(resp.status_code, resp.json().get('revision'))  # 200 + new revision
```

## Проект
- **URL**: `https://piper-next.artworks.ai`
- **Project ID**: `d2911d10bb`
- **Token**: `PIPER_TOKEN` из `.env`

## Локальный скрипт

`build_v6_full.py` — собирает pipeline_v2.json с обновлённым lgbm_evaluate + merged labels.  
Для деплоя запустить блок из этого файла или использовать скрипт ниже.

## Быстрый деплой lgbm_evaluate

```bash
cd /sessions/zealous-friendly-einstein/mnt/piper-moderate
python3 -c "
import json, os, httpx
from dotenv import load_dotenv
load_dotenv('.env')
TOKEN = os.getenv('PIPER_TOKEN')
hdr = {'User-Token': TOKEN, 'Content-Type': 'application/json'}
BASE = 'https://piper-next.artworks.ai/api'
PROJECT = 'd2911d10bb'

live = httpx.get(f'{BASE}/projects/{PROJECT}', headers=hdr, timeout=15).json()
revision = live['revision']
pipeline_live = json.loads(live['pipeline']) if isinstance(live['pipeline'], str) else live['pipeline']
old_script = pipeline_live['nodes']['lgbm_evaluate']['script']

p_v2 = json.loads(open('/sessions/zealous-friendly-einstein/pipeline_v2.json').read())
pipeline_local = json.loads(p_v2['pipeline']) if isinstance(p_v2['pipeline'], str) else p_v2['pipeline']
new_script = pipeline_local['nodes']['lgbm_evaluate']['script']

delta = {'pipeline': {'nodes': {'lgbm_evaluate': {'script': [old_script, new_script]}}}}
r = httpx.patch(f'{BASE}/projects/{PROJECT}/patch/{revision}', headers=hdr, content=json.dumps(delta), timeout=30)
print(r.status_code, r.json().get('revision'))
"
```

## Процедура: получение q3 возраста для новых Grafana-изображений

При добавлении новых изображений через `export_disagree.py` нужно **временно включить ноду Ask Qwen3** для получения возраста q3. Без этого галерея покажет только fd (face_detect) без описания и без точного возраста.

### Шаги

**1. Включить только qwen3 в Prepare params:**
```python
# Установить providers_e0.default = 'qwen3' в пайплайне
delta = {
    'pipeline': {
        'nodes': {
            'prepare_params': {
                'inputs': {
                    'providers': {
                        'default': ['qwen3']   # ADD — ключ default ранее отсутствовал
                    }
                }
            }
        }
    }
}
```
(Если default уже был: `'default': [old_value, 'qwen3']` — замена)

**2. Запустить скрипт для прогона новых изображений:**
```bash
cd /sessions/zealous-friendly-einstein/mnt/piper-moderate
# run_qwen3_age.py — запускает pipeline с providers_e0=['qwen3'] явно в launch
python3 /tmp/run_qwen3_age.py
```

**Важно:** скрипт при запуске передаёт `providers_e0: ['qwen3']` в launch-параметрах явно.  
Pipeline-level input называется **`providers_e0`** (не `providers`).  
Тестовый запуск: `{'inputs': {'image': url, 'providers_e0': ['qwen3']}}` → возвращает `qwen3_details`.

**3. Вернуть siglip2 + face_detect:**
```python
# Убрать default (если его не было) или вернуть на прежнее значение
delta = {
    'pipeline': {
        'nodes': {
            'prepare_params': {
                'inputs': {
                    'providers': {
                        'default': ['qwen3', None]   # DELETE — убрать добавленный default
                    }
                }
            }
        }
    }
}
```
После восстановления — нода вернётся к стандартному поведению (`['siglip2', 'hive']` по умолчанию в скрипте).

### Структура qwen3_result в пул-файле
```json
{
  "label": "teen",
  "faces": [{"ageFrom": 15, "ageTo": 17}],
  "description": "A young woman...",
  "underage": true,
  "status": "BLOCK",
  "processed_at": "2026-05-20T..."
}
```

### Структура piper_result.face_detect_result
```json
{"ageFrom": 10, "ageTo": 19, "gender": "unknown", "race": "white", "emotion": "neutral"}
```
Face detect данные должны лежать в `piper_result.face_detect_result`, **не** в `qwen3_result`.

---

## Авто-разметка новых карточек (логика)

При добавлении новых изображений из Grafana проставлять label по следующей логике:

- `siglip2_labels` **не содержит** `underage` (статус ok) → **adult**
- `siglip2_labels` **содержит** `underage` → смотреть на `fd.ageFrom`:
  - `ageFrom` от 3 до 9 включительно → **child**
  - иначе → **teen**

Пользователь всё равно перепроверяет вручную, это лишь предварительная разметка для удобства.

---

## Заметки по gallery.db

- **ls_images** — данные Label Studio. Таблица была удалена намеренно (не повреждение), не пытаться восстанавливать.
- **grafana_pool** — основная таблица, строится из disagree_pool.json.

## История деплоев

| Дата       | Ревизия от  | Ревизия до   | Что изменено                                    |
|------------|-------------|--------------|------------------------------------------------|
| 2026-05-20 | 6357b45756  | bfe2beb1df   | lgbm_evaluate V6 (AUC=0.8871, thr=0.55), +23 tags |
