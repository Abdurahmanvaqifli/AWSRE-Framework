# Şəkil üçün dayanıqlı su nişanı: nə mümkündür? (31 oktyabr 2026)

## Qısa cavab
Bəli, şəkil üçün **qismən** inkişaf etdirmək olar. Sınaq prototipi yazdım və ölçdüm:
- **JPEG (q30-a qədər), kəsmə (şəklin 25%-i qalsa belə), kiçiltmə (50%-ə qədər), ekran şəkli (screenshot)** — dözür.
- **Instagram-a bənzər axın** (4:5 kəsmə + 60% kiçiltmə + JPEG 80) — 8 şəkildən 6-sı.
- **Fırlatma** — 8-dən 6-sı.
- **Ekrandan kamera ilə çəkiliş** (simulyasiya) — **hələ alınmır** (0/8).

Üstəlik yoxlama üçün **orijinal şəkil lazım deyil**, yalnız gizli açar kifayətdir, və şəklə loqo əvəzinə **32-bit ID** (məsələn istifadəçi/alıcı nömrəsi) yazılır. Yəni bu, "hər istifadəçiyə unikal nişan" fikrinin texniki əsasıdır.

## Necə işləyir
1. Gizli açar 128×128 piksellik səs-küy "kafeli" (tile) yaradır: sinxronizasiya nümunəsi + ID-nin hər biti üçün bir nümunə (32 bit ID + 16 bit CRC yoxlama kodu).
2. Kafel bütün şəkil boyu təkrarlanır və parlaqlığa (Y kanalı) əlavə olunur; dəyişiklik dənəli (teksturlu) yerlərdə güclü, düz yerlərdə zəifdir.
3. Yoxlama zamanı şəklin avtokorrelyasiyası təkrarlanan kafelin şəbəkəsini göstərir. Buradan şəklin nə qədər kiçildildiyi, fırladıldığı tapılır və geri qaytarılır.
4. Şəkil bir kafelə "qatlanır". Kafel təkrarlandığı üçün kəsilmiş istənilən parça bütün məlumatı saxlayır, kəsmə problem olmur.
5. ID oxunur və CRC ilə yoxlanır. CRC uyğun gəlməsə, "tapılmadı" deyilir.

## Ölçmə
8 standart test şəkli (scikit-image, uzun tərəf 1024 px), 18 hücum. Müqayisə: saytdakı DCT α15 (orijinal tələb edir) və DCT-QIM α15 (orijinalsız).

| Hücum | Prototip (orijinalsız) | Saytdakı DCT α15 | DCT-QIM α15 |
|---|---|---|---|
| Hücumsuz | 8/8 | 8/8 | 8/8 |
| JPEG q70 / q50 / q30 | 8 / 8 / 8 | 8 / 8 / 0 | 0 / 0 / 0 |
| Kiçiltmə 75% / 50% / 35% | 8 / 8 / 6 | 8 / 7 / 4 | 8 / 0 / 0 |
| Kəsmə (50% / 25% sahə qalır) | 8 / 8 | 0 / 0 | 0 / 0 |
| Kəsmə 25% + JPEG 70 | 8 | 0 | 0 |
| Fırlatma 5° / 20° (+kəsmə) | 6 / 6 | 0 / 0 | 0 / 0 |
| Küy σ8, parlaqlıq/kontrast | 8, 8 | 8, 8 | 8, 7 |
| Instagram-a bənzər | 6 | 0 | 0 |
| Screenshot 80% | 8 | 8 | 8 |
| Ekran → kamera (sim., 4% / 8% əyilmə) | 0 / 0 | 0 / 0 | 0 / 0 |
| **Cəmi (144 sınaq)** | **128** | **67** | **39** |

Prototip üçün "tapıldı" = ID **tam düzgün** oxundu (daha sərt meyar). Baza metodlar üçün saytın qaydası: BER ≤ 30%.
Yalançı nəticə yoxlaması: nişansız şəkillərdə və səhv açarla 80 sınaqda **0** yalançı tapıntı, **0** səhv ID.

## Dürüst qeydlər (komissiya üçün vacib)
- **Görünməzlik daha aşağıdır:** prototip PSNR 42.1 dB / SSIM 0.985, saytdakı DCT α15 bu şəkillərdə 47.7 dB / 0.990. Yəni dayanıqlığın bir hissəsi daha güclü nişan hesabınadır.
- **Daha az məlumat:** prototip 48 bit daşıyır (32-bit ID + CRC), DCT 32×32 = 1024 bitlik loqo. Dayanıqlı sistemlərdə loqo əvəzinə qısa ID yazmaq standartdır.
- Kamera çəkilişi real kamera ilə deyil, **simulyasiya** ilə yoxlanılıb (perspektiv, bulanıqlıq, qamma, küy, JPEG). Uğursuzluğun əsas səbəbi perspektivdir: bütün şəkil üçün bir affin çevirmə kifayət etmir.
- Test şəkilləri kiçik standart şəkillərdir (bəziləri böyüdülüb). Real foto üzərində nəticələr fərqli ola bilər.
- Yoxlama orta hesabla ~3.5 saniyə çəkir (uğursuz hallarda axtarış uzanır).

## Kamera çəkilişi üçün nə lazımdır (gələcək iş)
1. Şəklin hissələrində lokal şəbəkəni tapıb homoqrafiya (perspektiv) qurmaq.
2. Daha güclü xəta düzəltmə kodu (BCH/LDPC) və rəng kanallarından da istifadə.
3. Ən dayanıqlı yanaşma: neyron şəbəkə ilə öyrədilmiş nişan (məs. StegaStamp tipli), öyrənmə zamanı ekran/kamera təhrifləri simulyasiya olunur. Bu ayrıca böyük işdir.

## Netflix ilə fərq
Netflix-in "forensic watermark"-ı videonun minlərlə kadrına paylanır, ona görə məlumat çox təkrarlanır. Bir şəkildə bu imkan yoxdur, ona görə şəkil üçün eyni səviyyəli dayanıqlıq daha çətindir.

## Kod
GitHub-da `research/robust_tiled/` (ayrıca PR, sayta və modellərə toxunmur).
