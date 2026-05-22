# Workflow Guide

> Актуально на 2026-05-17.

---

## ⚠️ Ключевая концепция: сырые данные vs. эталон галереи

### Два типа данных — нельзя путать

| Тип | Источник | Что означает |
|-----|----------|--------------|
| **Сырые данные** | `age_group_v2_N.json` / прогоны Piper | Снапшот текущего скоринга SigLIP2 с текущими тегами. Могут быть неточными — зависят от качества тегов на момент прогона. |
| **Эталон** (ground truth) | `qwen3_age_results.json` + `data/disagree_pool.json` | Ручная разметка пользователя в галерее (child/teen/adult). Это правда — на неё равняемся. |

### Цель проекта

Настроить теги `data/tags.json` так, чтобы результаты Piper-скоринга **совпадали с ручной разметкой галереи** в пределах допустимой погрешности:

| Категория | Метрика | Цель |
|-----------|---------|------|
| child | recall = blocked_child / total_child | **≥ 98%** |
| teen | recall = blocked_teen / total_teen | **≥ 80%** |
| adult | FPR = blocked_adult / total_adult | **≤ 20%** |

### Источники эталонных меток

**Label Studio** (`qwen3_age_results.json`):
- Ключ словаря = `str(task_id)` (напр. `'19490'`)
- Галерейный ID = `ls_{task_id}` (напр. `ls_19490`)
- Возраст: `ageFrom ≤ 14 → child`, `≤ 17 → teen`, `≥ 18 → adult`

**Grafana** (`data/disagree_pool.json`):
- Ключ словаря = UUID (напр. `019e34e8-...`)
- Галерейный ID = `dg_{UUID}`
- Метка: поле `label` (`child`/`teen`/`adult`), source `human`

### Маппинг ID из групп → эталонные метки

| ID в group JSON | Как найти метку |
|-----------------|-----------------|
| `qwen_19490` | → task_id=19490 → `qwen3_age_results['19490']['age']` |
| `ls_823` | → task_id=823 → **не в `qwen3_age_results`** — это удалённые/исключённые элементы (417 шт.), игнорировать |
| `dg_019e34e8-...` | → UUID → `disagree_pool['019e34e8-...']['label']` |

### Актуальный эталонный датасет (2026-05-17)

Всего **3296 изображений** с ручными метками:

| Метка | Кол-во | Знаменатель для метрики |
|-------|--------|------------------------|
| child | 424 | recall: blocked / 424 |
| teen | 1159 | recall: blocked / 1159 |
| adult | 1713 | FPR: blocked / 1713 |

*Примечание: галерея показывает ~3404 (child 432, teen 1214, adult 1758) — разница ~108 шт. из-за того, что `qwen3_age_results.json` не содержит самые последние добавленные LS-элементы. При обновлении файла цифры сойдутся.*

### Правило «свежий прогон» для оценки точности

При оценке метрик всегда использовать **свежий прогон** через текущую Piper (с актуальными тегами и LGBM v4, threshold=0.80). Хранящиеся в group JSON скоры — это снапшоты старых конфигураций, они **не годятся** для расчёта текущих метрик.

Исключение — если прогон был сделан сегодня с текущей конфигурацией (можно проверить по `revision` проекта в Piper).

Текущая конфигурация Piper (2026-05-17):
- LGBM v4: 100 деревьев, 15 листьев, AUC=0.977, порог=0.80
- SigLIP: теги v9 (690 тегов), confidence_threshold=0.72
- Правило блокировки: `lgbm_score ≥ 0.80 OR minor ≥ 0.72`

---

## Ресурсы и доступы

Все токены, логины и пароли хранятся в [`credentials.md`](credentials.md).

| Ресурс | URL |
|--------|-----|
| Piper Studio | https://piper-next.artworks.ai/en/projects/d2911d10bb |
| Тестовый датасет | https://mod.artworks.ai/# |
| Результаты лончей | https://piper-next.artworks.ai/en/projects/d2911d10bb/launches |

---

## Метрики симулятора

| Метрика | Формула | Цель |
|---------|---------|------|
| Accuracy | TP / (TP + FN) | Recall — доля правильно найденных positives |
| Error | FP / (FP + TN) | FPR — доля ложных срабатываний на negatives |

**Failed+** = модель пропустила positive-сэмпл → улучшить теги  
**Failed−** = модель ложно сработала на negative → сузить теги или повысить порог

**Unsure** items исключаются из расчёта Error, но входят в Total.

---

## Основной цикл улучшения тегов

### Шаг 1 — Обновить теги в `data/tags.json`

Редактировать файл вручную или через Claude. Ключевые принципы:
- Теги должны быть **универсальными** — описывать визуальный класс, а не конкретное изображение
- Для каждой категории достаточно **8–12 разнообразных тегов** (не дублировать похожие сцены)
- Описания длиной **до ~28 слов** (лимит токенизатора SigLIP-2 ≈ 64 wordpiece)
- Не использовать отрицания (`not anime`, `not illustrated`) — SigLIP их игнорирует
- Ключ тега должен начинаться с префикса категории: `underage_`, `ebony_`, `asian_` и т.д.

### Шаг 2 — Загрузить теги в Piper (PATCH API + инвалидация кэша)

**Важно:** простой патч `inputs.labels.default` не инвалидирует SigLIP PaaS-кэш.
Необходимо **одновременно изменить скрипт ноды** (добавить версионный комментарий),
чтобы пересчитался `sign` ноды — это и есть ключ кэша.

```python
import json, subprocess, tempfile, os, re

TOKEN = "<PIPER_TOKEN из credentials.md>"
HOST  = "piper-next.artworks.ai"
PROJECT = "d2911d10bb"

with open('data/tags.json') as f:
    tags = json.load(f)

# Получаем текущее состояние проекта
res = subprocess.run(['curl','-s','-H',f'User-Token: {TOKEN}',
    f'https://{HOST}/api/projects/{PROJECT}'], capture_output=True, text=True)
proj = json.loads(res.stdout)
rev  = proj['revision']
node = proj['pipeline']['nodes']['ask_siglip2']
old_script  = node['script']
old_default = node['inputs']['labels']['default']

# Инкрементируем версию в комментарии скрипта (инвалидирует кэш!)
old_ver    = re.search(r'// v(\d+)', old_script)
new_ver    = (int(old_ver.group(1)) + 1) if old_ver else 1
new_script = re.sub(r'// v\d+\n?', '', old_script).rstrip() + f"\n// v{new_ver}\n"

delta = {"pipeline": {"nodes": {"ask_siglip2": {
    "script": [old_script, new_script],
    "inputs": {"labels": {"default": [old_default, json.dumps(tags, ensure_ascii=False, indent=2)]}}
}}}}

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(delta, f, ensure_ascii=False); tmp = f.name

subprocess.run(['curl','-s','-X','PATCH',
    '-H',f'User-Token: {TOKEN}','-H','Content-Type: application/json',
    '-d',f'@{tmp}', f'https://{HOST}/api/projects/{PROJECT}/patch/{rev}'])
os.unlink(tmp)
```

Можно также обновить пороги в ноде `siglip_config` (тот же механизм PATCH):

```python
# Например, изменить label_threshold с 0.15 на 0.10
old_cfg_script = proj['pipeline']['nodes']['siglip_config']['script']
new_cfg_script = old_cfg_script.replace('label_threshold:          0.15',
                                        'label_threshold:          0.10')
delta_cfg = {"pipeline": {"nodes": {"siglip_config": {
    "script": [old_cfg_script, new_cfg_script]
}}}}
# ... аналогичный PATCH запрос
```

### Шаг 3 — Запустить тестовые прогоны

```python
# Загружаем снапшот с тестовыми сэмплами
with open('data/snapshot.json') as f:
    data = json.load(f)

samples = [r for r in data
           if r.get('category') == 'underage'   # нужная категория
           and r.get('media')
           and not r.get('uncertain')
           and r.get('variant') in ('positive','negative')]

# Запускаем прогоны (по 30 за раз — shell-таймаут ~90 сек)
launched = []
for s in samples[:30]:
    res = subprocess.run(['curl','-s','--max-time','9','-X','POST',
        '-H',f'User-Token: {TOKEN}','-H','Content-Type: application/json',
        '-d', json.dumps({"inputs":{"image":s['media'],"providers":["siglip2"]}}),
        f'https://{HOST}/api/projects/{PROJECT}/launch'],
        capture_output=True, text=True, timeout=11)
    try:
        resp = json.loads(res.stdout)
        launched.append({'sample_id':s['id'],'run_id':resp['_id'],
                         'category':s['category'],'variant':s.get('variant')})
    except: pass
```

> **Ограничение:** bash-сессия завершается примерно через 90 секунд.
> Запускайте не более 30 сэмплов за раз, разбивая на батчи.

### Шаг 4 — Собрать результаты

```python
# После ~45 секунд ожидания опрашиваем состояние прогонов
completed = {}
for r in launched:
    res = subprocess.run(['curl','-s','--max-time','6',
        '-H',f'User-Token: {TOKEN}',
        f'https://{HOST}/api/launches/{r["run_id"]}/state'],
        capture_output=True, text=True, timeout=8)
    try:
        st   = json.loads(res.stdout)
        outs = st.get('outputs') or {}
        if outs.get('siglip2_details'):
            completed[str(r['sample_id'])] = {'outputs': outs, **r}
    except: pass
```

Результаты: `outputs.siglip2_details.<category>.score` — итоговый скор,
`outputs.siglip2_details.<category>.labels` — топ-теги выше `min_label_display_score`.

### Шаг 5 — Анализ через Gemini (для визуального аудита failing-сэмплов)

Для изображений с явным контентом используй OpenRouter + Gemini:

```python
import base64, json, urllib.request

KEY = "<OPENROUTER_API_KEY из credentials.md>"

def analyze_image(path, prompt):
    with open(path,'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({
        "model": "google/gemini-3.1-pro",
        "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}},
            {"type":"text","text":prompt}
        ]}],
        "max_tokens": 200
    }).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())['choices'][0]['message']['content']
```

> Использовать `Read` tool Claude для просмотра NSFW-изображений нельзя —
> Anthropic API возвращает 400. Gemini через OpenRouter лояльнее к такому контенту.

### Шаг 6 — Симулятор

Открыть `simulator/index.html` в браузере для:
- Визуального просмотра результатов по всем категориям
- Подбора оптимальных порогов (слайдеры в сайдбаре)
- Сравнения разных конфигурационных профилей

Симулятор читает `models.siglip2.details` из снапшота — убедись, что снапшот
актуален (загружен из `mod.artworks.ai` или собран из API-прогонов).

---

## Формула комбинированного скора (SigLIP combineScores)

```
score = 1 − ∏(1 − score_i)  для всех тегов категории
```

Несколько тегов с низкими скорами дают значимый суммарный сигнал:
- 5 тегов по 0.03 → combined ≈ 0.14
- 3 тега по 0.05 → combined ≈ 0.14

---

## Типичные проблемы

### Теги загружены, но результаты не изменились

Кэш SigLIP PaaS не инвалидирован. Убедись что при PATCH изменился `sign`
ноды `ask_siglip2` — это происходит только если изменён **скрипт** (не только labels).

### `siglip2_details` в ответе пустой

Прогон ещё не завершён. Подожди 30–60 секунд и опроси снова.
Для fresh-прогонов (новый sign) SigLIP тратит 10–30 сек на изображение.

### Recall низкий при нулевом FPR

Порог `label_threshold` в `siglip_config` может быть слишком высоким.
Текущее значение для этнических категорий: **0.10** (было 0.15).
Снижение порога увеличивает recall, но потенциально повышает FPR.

### Описания тегов слишком длинные

SigLIP-2 токенизирует описания с лимитом ~64 wordpiece. Длинные описания
обрезаются, что снижает скор. Держи описания до **~28 слов** или проверяй
токенизацию перед загрузкой.

### Негативные квалификаторы не работают

Фраза `"not anime"` или `"not illustrated"` в описании тега не уменьшает скор
для аниме-изображений — SigLIP игнорирует отрицания. Вместо этого перепиши
описание с акцентом на **позитивные визуальные признаки** нужного класса.
