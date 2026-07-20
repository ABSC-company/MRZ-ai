# Сканер MRZ и полей документа, удостоверяющего личность

Сервис FastAPI для документов, удостоверяющих личность. Сервис сначала
выполняет обнаружение, распознавание, парсинг и валидацию MRZ. Если MRZ
отсутствует или невалиден, резервный детектор RT-DETR (9 классов) находит
визуальные поля личности и оставляет только бокс с наивысшим score для
каждого класса.

## Схема обработки

```text
Загрузка -> Проверка безопасности -> MRZ-пайплайн -> Валидация ICAO
                                            |
                                            +-- Валиден --> Ответ с распарсенным MRZ
                                            |
                                            +-- Отсутствует/невалиден --> RT-DETR -> OCR
```

API не сохраняет загруженные документы. Продакшн-логи содержат ID запроса,
формат, размер в байтах, тайминг, источник и статус; имена файлов, текст MRZ
и распознанные персональные поля не логируются.

## Структура проекта

```text
MRZ Project/
  app/
    main.py
    api/v1/endpoints/mrz.py
    api/v1/router.py
    core/config.py
    core/logging.py
    core/security.py
    models/detection.py
    models/recognition.py
    services/document.py
    services/fallback.py
    services/mrz_pipeline.py
    services/preprocessing.py
    services/postprocessing.py
    services/validation.py
    schemas/request.py
    schemas/response.py
    utils/image.py
    utils/exceptions.py
  configs/cards_rtdetrv2_r18.yml
  weights/
    unet_resnet34.pth
    cards_rtdetrv2_r18_best.pth
  vendor/
    MRZScanner/
    rtdetrv2_pytorch/
  deploy/Caddyfile
  secrets/хранение конфиденциальных данных 
  tests/проверка корректности разбора форматов
  Dockerfile
  docker-compose.yml
  docker-compose.cpu.yml
  requirements.txt
```


## Локальный запуск на Python

Поддерживаемый интерпретатор — Python 3.11.

```powershell
cd "D:\ML\MRZ Project"
.\.venv\Scripts\activate
python -m pip install -e .\vendor\MRZScanner
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m uvicorn app.main:app --host 127.0.0.1 --port 5000
```

По умолчанию для разработки API-ключ не требуется, а Swagger доступен:

```text
http://127.0.0.1:5000/docs
```

## API

Основные эндпоинты:

```text
POST /api/v1/mrz/recognize
GET  /api/v1/health
GET  /api/v1/models/info
```

`POST /scan` и `GET /health` сохранены как скрытые алиасы для обратной
совместимости. Эндпоинты scan и model-info требуют заголовок `X-API-Key`,
если включена переменная `API_KEY_REQUIRED=1`.

Локальный запрос:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/api/v1/mrz/recognize" `
  -H "accept: application/json" `
  -F "file=@D:\test\1.png"
```

Поддерживаемые форматы загрузки: JPEG, PNG, TIFF, PDF, BMP, WebP, HEIC и HEIF.
Проверяются расширение, MIME-тип, сигнатура файла, результат декодирования,
количество пикселей изображения и количество страниц документа. Значения по
умолчанию — 20 МБ, 60 миллионов пикселей на страницу и 10 страниц;
настраиваются через переменные окружения.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Используйте `production` в Docker |
| `ENABLE_DOCS` | включено вне production | Swagger/OpenAPI |
| `API_KEY_REQUIRED` | включено в production | Защита scan/model-info |
| `API_KEY_FILE` | `/run/secrets/mrz_api_key` | Путь к Docker-секрету |
| `MAX_UPLOAD_MB` | `20` | Максимальный размер загружаемого файла |
| `MAX_IMAGE_PIXELS` | `60000000` | Лимит на декомпрессию/пиксели |
| `MAX_DOCUMENT_PAGES` | `10` | Лимит страниц PDF/TIFF |
| `MRZ_DEVICE` | `auto` | `cuda`, `cpu` или `auto` |
| `FALLBACK_DEVICE` | `auto` | Устройство для RT-DETR/OCR |
| `FALLBACK_THRESHOLD` | `0.40` | Порог уверенности детекции |
| `TRUSTED_HOSTS` | `*` | Список допустимых значений Host через запятую |

Все существующие переменные ориентации MRZ, обрезки и EasyOCR по-прежнему
доступны в `.env.example`.

## Предварительные требования для Docker

Для развёртывания по умолчанию (GPU) установите:

1. Docker Desktop с бэкендом WSL2.
2. Актуальный драйвер NVIDIA с поддержкой WSL.
3. Docker Compose v2 с поддержкой GPU.

Перед сборкой большого образа убедитесь, что `docker compose version`
работает и Docker может использовать GPU NVIDIA.

Следующие файлы должны существовать на хосте и монтируются только для чтения:

```text
weights/unet_resnet34.pth
weights/cards_rtdetrv2_r18_best.pth
```

Веса и секреты намеренно исключены из образа и обычного Git.

## Развёртывание в продакшн через Docker

### 1. Создание конфигурации

```powershell
cd "D:\ML\MRZ Project"
Copy-Item .env.example .env
notepad .env
```

Укажите в `DOMAIN` реальное DNS-имя, указывающее на сервер. Порты 80 и 443
должны быть доступны на этом хосте, чтобы Caddy мог получить и продлевать
публичный сертификат. Для локального теста используйте `DOMAIN=localhost`;
Caddy выпустит локальный сертификат, который Windows не будет доверять
автоматически.

### 2. Генерация API-ключа

```powershell
New-Item -ItemType Directory -Path .\secrets -Force | Out-Null
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$key = [BitConverter]::ToString($bytes).Replace("-", "").ToLowerInvariant()
[System.IO.File]::WriteAllText((Join-Path $PWD "secrets\api_key.txt"), $key)
```

Никогда не коммитьте `secrets/api_key.txt` или `.env`.

### 3. Проверка и сборка

```powershell
docker compose config
docker compose build
```

Первая сборка устанавливает CUDA-версию PyTorch и загружает модели EasyOCR в
образ, чтобы read-only рантайм не скачивал файлы при запуске.

### 4. Запуск

```powershell
docker compose up -d
docker compose ps
docker compose logs -f mrz-api
```

```
https://localhost/docs
```

Контейнер API не публикуется напрямую. Caddy — единственный публичный
сервис, он пересылает HTTPS-запросы через внутреннюю сеть Docker.

### 5. Проверка

```powershell
$key = (Get-Content .\secrets\api_key.txt -Raw).Trim()
$domain = ((Get-Content .env | Where-Object { $_ -match '^DOMAIN=' }) -replace '^DOMAIN=', '').Trim()
curl.exe "https://$domain/api/v1/health"
curl.exe -X POST "https://$domain/api/v1/mrz/recognize" `
  -H "X-API-Key: $key" `
  -F "file=@D:\test\1.png"
```

Для `DOMAIN=localhost` добавляйте `-k` только во время локального теста
сертификата.

## Вариант для CPU (Docker)

```powershell
docker compose -f docker-compose.yml -f docker-compose.cpu.yml build
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Инференс на CPU поддерживается, но существенно медленнее, особенно для
RT-DETR и EasyOCR.

## Обслуживание

```powershell
docker compose logs --tail 200 mrz-api
docker compose restart mrz-api
docker compose build --pull
docker compose up -d --remove-orphans
docker compose down
```

Состояние сертификатов Caddy хранится в именованных томах. `docker compose
down` их сохраняет; `docker compose down -v` удаляет и не должен
использоваться при обычном обновлении.

## Свойства безопасности

- лимиты на байты, пиксели и страницы загружаемого файла применяются как
  Caddy, так и приложением;
- содержимое проверяется по сигнатуре и реальным декодером
  изображений/PDF;
- загрузки закрываются после каждого успеха или ошибки, `/tmp` —
  файловая система в памяти с ограничением размера;
- контейнер API работает под UID 10001, сбрасывает Linux-capabilities,
  запрещает повышение привилегий и имеет корневую файловую систему только
  для чтения;
- веса моделей монтируются только для чтения и не копируются в образ;
- продакшн API-ключи используют Docker secrets и сравнение за постоянное
  время;
- распознанные поля и полный MRZ возвращаются только авторизованному
  вызывающему и не записываются в логи приложения;
- логи Docker ротируются при 10 МБ, хранится три файла.

На уровне хоста ограничьте доступ к `weights/`, `secrets/`, Docker Desktop и
директории `archive/`. Архив может содержать старые локальные образцы и не
должен публиковаться.

## Тесты

```powershell
python -m unittest discover -s tests -v
```

Тесты покрывают парсинг TD1/TD3, страновые профили, правила персонального
номера, выбор единственного лучшего результата в fallback-режиме, лимиты
размера потока, несоответствие формата и конфигурацию API-ключа для
продакшна.
