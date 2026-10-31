# Dayanıqlı (özü sinxronlaşan) su nişanı: prototip

Saytda və AWSRE tövsiyə modellərində **istifadə olunmur**. "Gələcək iş" sualı üçün tədqiqat prototipidir:
şəkil üçün su nişanı JPEG, kəsmə, kiçiltmə və ekran/kamera çəkilişinə dözə bilərmi?

- `robust_wm.py`: 128×128 kafel (tile), sinxronizasiya nümunəsi + 32-bit ID + CRC-16, yalnız açarla (orijinalsız) yoxlama.
- `benchmark_robust.py`: 8 şəkil × 18 hücum, saytdakı DCT α15 və DCT-QIM α15 ilə müqayisə.
- `results.json`: ölçmənin xam nəticələri.

İşə salmaq (repo kökündən): `python research/robust_tiled/benchmark_robust.py` (4 nüvədə ~10 dəq).
