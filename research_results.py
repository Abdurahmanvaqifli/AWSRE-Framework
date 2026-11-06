"""
AWSRE Framework
"Research results" tab of the Home page.

Summarises the final frozen experiment run in Colab (results in the
experiments/ folder on Drive, FINAL_FROZEN_UNSEEN_TEST). The numbers are
copied here as constants so the public site does not need the large
result files. Reported honestly: what is significant and what is not.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.recommender import MODEL_SUMMARY, TRAINED_METHODS
from i18n import t


def _significance(p: float) -> str:
    return t("Yes", "Bəli") if p < 0.05 else t("No", "Xeyr")


def _vs_fixed_table() -> pd.DataFrame:
    # metric, AWSRE, fixed DCT a=15, difference, better for AWSRE, Holm p
    rows = [
        ("PSNR (dB)", "PSNR (dB)", "42.404", "41.958", "+0.446", True, 1.1e-13),
        ("SSIM", "SSIM", "0.97761", "0.97636", "+0.00125", True, 1.1e-13),
        ("BER (wrong bits after attacks)", "BER (hücumdan sonra səhv bitlər)",
         "0.15456", "0.15192", "+0.00264", False, 1.3e-7),
        ("Distance to ideal point (excess)", "İdeal nöqtəyə məsafə (artıq)",
         "0.01876", "0.02004", "−0.00128", True, 0.263),
        ("Pareto hit rate", "Pareto uğur faizi", "94.69%", "92.81%", "+1.88 pt", True, 0.591),
    ]
    return pd.DataFrame(
        [
            {
                t("Metric", "Metrik"): t(en, az),
                "AWSRE": awsre,
                t("Fixed DCT α=15", "Sabit DCT α=15"): fixed,
                t("Difference", "Fərq"): diff,
                t("Better for AWSRE?", "AWSRE üçün daha yaxşı?"): t("Yes", "Bəli") if better else t("No", "Xeyr"),
                t("Holm p", "Holm p"): f"{p:.2g}",
                t("Significant?", "Əhəmiyyətli?"): _significance(p),
            }
            for en, az, awsre, fixed, diff, better, p in rows
        ]
    )


def _vs_other_baselines_table() -> pd.DataFrame:
    rows = [
        ("AWSRE", "AWSRE", "94.69%", "42.40", "0.1546", "0.0188"),
        ("Fixed DWT α=25", "Sabit DWT α=25", "84.38%", "43.64", "0.1732", "0.0726"),
        ("Fixed DCT-SVD α=25", "Sabit DCT-SVD α=25", "1.25%", "–", "0.2290", "0.864"),
        ("Random choice", "Təsadüfi seçim", "59.38%", "43.13", "0.1956", "0.495"),
    ]
    return pd.DataFrame(
        [
            {
                t("Strategy", "Strategiya"): t(en, az),
                t("Pareto hit rate", "Pareto uğur faizi"): pareto,
                "PSNR (dB)": psnr,
                "BER": ber,
                t("Distance to ideal (excess)", "İdeala məsafə (artıq)"): utopia,
            }
            for en, az, pareto, psnr, ber, utopia in rows
        ]
    )


def _literature_table() -> pd.DataFrame:
    rows = [
        ("AWSRE intelligent (DCT α=20)", "AWSRE intellektual (DCT α=20)", "40.193", "0.980365", "0.1126", "0.0088", "7"),
        ("AWSRE fixed (DCT α=15)", "AWSRE sabit (DCT α=15)", "42.696", "0.988119", "0.1222", "0.0137", "5"),
        ("Abdul Karim, SVD + PSO", "Abdul Karim, SVD + PSO", "45.625", "0.993949", "0.1357", "0.0703", "1"),
        ("Pasagadugula-Kota 2026 (adapted)", "Pasagadugula-Kota 2026 (uyğunlaşdırılmış)", "56.316", "0.999593", "0.1631", "0.0117", "8"),
        ("Dong 2024 (adapted) *", "Dong 2024 (uyğunlaşdırılmış) *", "47.086", "0.999509", "0.0756", "–", "–"),
    ]
    return pd.DataFrame(
        [
            {
                t("Method", "Metod"): t(en, az),
                "PSNR (dB)": psnr,
                "SSIM": ssim,
                t("Mean BER", "Orta BER"): mean_ber,
                t("Median BER", "Median BER"): median_ber,
                t("Attacks with BER 0", "BER 0 olan hücumlar"): zero,
            }
            for en, az, psnr, ssim, mean_ber, median_ber, zero in rows
        ]
    )


def _pct(value) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def show_current_model_results() -> None:
    """Test-set results of the models the site is using now, read from
    model_summary.json written when they were trained. Shown only for the
    5-method models; nothing here is typed in by hand."""
    summary = MODEL_SUMMARY
    if not summary:
        return
    rec = summary.get("recommendation") or {}
    r2 = summary.get("r2") or {}
    best = rec.get("best_fixed_strategy") or {}
    st.subheader(t("Models in use: {k} methods", "İstifadə olunan modellər: {k} metod").format(k=len(summary["methods"])))
    st.markdown(t(
        "The site now chooses among **{n} strategies** ({m}; α {a}). Trained on {rows:,} measurements "
        "from {hosts} host images; tested on host images never seen in training ({ctx} test contexts).",
        "Sayt indi **{n} strategiya** arasından seçir ({m}; α {a}). {hosts} əsas şəkil üzrə {rows:,} "
        "ölçmə ilə öyrədilib; öyrənmədə görülməmiş şəkillər üzərində test olunub ({ctx} test konteksti).",
    ).format(
        n=summary["strategies"], m=", ".join(summary["methods"]),
        a=", ".join(f"{a:g}" for a in summary["alphas"]),
        rows=summary["dataset_rows"], hosts=summary["hosts"], ctx=rec.get("contexts", 0),
    ))
    st.dataframe(pd.DataFrame([
        {t("Prediction", "Proqnoz"): "PSNR", "R²": f"{r2.get('clean_psnr', float('nan')):.3f}"},
        {t("Prediction", "Proqnoz"): "SSIM", "R²": f"{r2.get('clean_ssim', float('nan')):.3f}"},
        {t("Prediction", "Proqnoz"): t("BER after attacks (mean)", "Hücumdan sonra BER (orta)"),
         "R²": f"{r2.get('mean_attack_ber', float('nan')):.3f}"},
        {t("Prediction", "Proqnoz"): t("Embedding time", "Yerləşdirmə vaxtı"),
         "R²": f"{r2.get('embedding_time_seconds', float('nan')):.3f}"},
    ]), hide_index=True, use_container_width=True)
    st.dataframe(pd.DataFrame([
        {t("Choice", "Seçim"): t("AWSRE (all {k} methods)", "AWSRE (bütün {k} metod)").format(k=len(summary["methods"])),
         t("Pareto hit rate", "Pareto uğur faizi"): _pct(rec.get("pareto_hit_rate"))},
        {t("Choice", "Seçim"): t("Best fixed strategy: {m}, α = {a:g}", "Ən yaxşı sabit strategiya: {m}, α = {a:g}").format(
            m=best.get("method", "?"), a=float(best.get("alpha", 0))),
         t("Pareto hit rate", "Pareto uğur faizi"): _pct(best.get("pareto_hit_rate"))},
        {t("Choice", "Seçim"): t("AWSRE limited to the old 3 methods", "AWSRE yalnız köhnə 3 metodla"),
         t("Pareto hit rate", "Pareto uğur faizi"): _pct(rec.get("old3_only_pareto_hit_rate_within_all"))},
    ]), hide_index=True, use_container_width=True)
    shares = rec.get("awsre_pick_share_by_method") or {}
    st.caption(t(
        "Pareto hit = the chosen strategy is on the real (measured) Pareto front of all {n} strategies. "
        "With more strategies the front is larger, so these rates are not directly comparable with "
        "the dissertation's 3-method numbers below. How often AWSRE picks each method: {s}.",
        "Pareto uğuru = seçilən strategiya bütün {n} strategiyanın real (ölçülmüş) Pareto cəbhəsindədir. "
        "Strategiya çox olduqca cəbhə böyüyür, ona görə bu faizlər aşağıdakı dissertasiya "
        "(3 metod) rəqəmləri ilə birbaşa müqayisə olunmur. AWSRE hər metodu nə qədər seçir: {s}.",
    ).format(n=summary["strategies"], s=", ".join(f"{k} {100 * v:.0f}%" for k, v in shares.items())))
    st.divider()


def show_research_results() -> None:
    show_current_model_results()
    if MODEL_SUMMARY:
        st.markdown(t("### Dissertation run (3 methods, 15 strategies)",
                      "### Dissertasiya təcrübəsi (3 metod, 15 strategiya)"))
    st.subheader(t("Test on unseen data", "Görülməmiş verilənlər üzərində test"))
    st.markdown(
        t(
            "Final frozen test: **320 contexts** (160 host images) that the models never saw during "
            "training. AWSRE's per-image choice is compared with the best single fixed strategy "
            "(DCT, α = 15). Wilcoxon signed-rank test with Holm correction.",
            "Son, dondurulmuş test: modellərin öyrənmə zamanı heç görmədiyi **320 kontekst** "
            "(160 əsas şəkil). AWSRE-nin hər şəkil üçün seçimi ən yaxşı tək sabit strategiya "
            "(DCT, α = 15) ilə müqayisə olunur. Wilcoxon testi, Holm düzəlişi ilə.",
        )
    )
    st.dataframe(_vs_fixed_table(), hide_index=True, use_container_width=True)
    st.markdown(
        t(
            "**In short:** AWSRE gives significantly better image quality (PSNR, SSIM) at the cost of a "
            "very small but significant increase in BER. Its Pareto hit rate is higher (+1.88 points, "
            "95% CI −0.31 to +4.06), but that difference is **not** statistically significant.",
            "**Qısaca:** AWSRE şəkil keyfiyyətini (PSNR, SSIM) əhəmiyyətli dərəcədə yaxşılaşdırır, "
            "əvəzində BER çox az, amma əhəmiyyətli şəkildə artır. Pareto uğur faizi daha yüksəkdir "
            "(+1.88 bənd, 95% CI −0.31 … +4.06), lakin bu fərq statistik əhəmiyyətli **deyil**.",
        )
    )

    st.markdown(t("#### Against other strategies", "#### Digər strategiyalarla müqayisə"))
    st.dataframe(_vs_other_baselines_table(), hide_index=True, use_container_width=True)
    st.caption(
        t(
            "Paired over 160 hosts. AWSRE is significantly better than fixed DWT α=25, fixed DCT-SVD α=25 "
            "and random choice on Pareto hit rate, BER and distance to the ideal point. Fixed DWT α=25 "
            "and random choice have higher PSNR.",
            "160 əsas şəkil üzrə cütləşdirilmiş. AWSRE Pareto uğur faizi, BER və ideal nöqtəyə məsafə "
            "üzrə sabit DWT α=25, sabit DCT-SVD α=25 və təsadüfi seçimdən əhəmiyyətli dərəcədə yaxşıdır. "
            "Sabit DWT α=25 və təsadüfi seçimin PSNR-i daha yüksəkdir.",
        )
    )

    st.divider()
    st.subheader(t("Comparison with published methods", "Ədəbiyyatdakı metodlarla müqayisə"))
    st.info(
        t(
            "Descriptive only: **one host image**, 17 attacks. This is not a statistical test.",
            "Yalnız təsviri: **bir əsas şəkil**, 17 hücum. Bu statistik test deyil.",
        )
    )
    st.dataframe(_literature_table(), hide_index=True, use_container_width=True)
    st.caption(
        t(
            "* Dong 2024's extractor also returns BER 0 on the original, unwatermarked image "
            "(negative control), so its low BER is a false positive and does not mean it is better. "
            "AWSRE vs Pasagadugula-Kota per attack: 8 wins, 3 losses, 6 ties. Kumar 2026 was not "
            "reproduced (paper not available).",
            "* Dong 2024-ün çıxarıcısı su nişanı olmayan orijinal şəkildə də BER 0 verir "
            "(neqativ nəzarət), yəni onun aşağı BER-i yalançı müsbətdir və daha yaxşı olduğunu göstərmir. "
            "AWSRE vs Pasagadugula-Kota hücum üzrə: 8 qələbə, 3 məğlubiyyət, 6 bərabərlik. Kumar 2026 "
            "təkrarlanmadı (məqalə əlçatan deyil).",
        )
    )

    st.divider()
    show_limitations_and_future_work()


def show_limitations_and_future_work() -> None:
    st.subheader(t("Limitations and future work", "Məhdudiyyətlər və gələcək iş"))

    st.markdown(t("#### Limitations", "#### Məhdudiyyətlər"))
    st.markdown(
        t(
            "- **One strategy dominates.** Without the time criterion (MSE and BER only) AWSRE picks "
            "DCT α = 15 in 318 of 320 test contexts. The oracle, using real measurements instead of "
            "predictions, also picks DCT α = 15 in 84% of them, so this reflects the data, not a "
            "model bias. Adaptivity pays off mainly under blur and when a minimum PSNR is required.\n"
            "- **The methods AWSRE chooses from need the original photo to verify** (non-blind; "
            "DCT-QIM, when included, is the exception), "
            "and rotation or cropping destroys their mark. The separate robust mode (below) is "
            "verified without the original, but it is not part of the ML recommender.",
            "- **Bir strategiya üstünlük təşkil edir.** Vaxt meyarı olmadan (yalnız MSE və BER) AWSRE "
            "320 test kontekstindən 318-də DCT α = 15 seçir. Proqnoz əvəzinə real ölçmələrdən istifadə "
            "edən oracle da onların 84%-də DCT α = 15 seçir, yəni bu, modelin meyli deyil, verilənlərin "
            "özünü əks etdirir. Adaptivlik əsasən bulanıqlıqda və minimum PSNR tələb olunanda fayda verir.\n"
            "- **AWSRE-nin seçdiyi metodlar yoxlama üçün orijinal şəkil tələb edir** (kor olmayan; "
            "DCT-QIM, əgər daxildirsə, istisnadır), "
            "fırlatma və kəsmə isə onların nişanını məhv edir. Ayrıca dayanıqlı rejim (aşağıda) "
            "orijinalsız yoxlanılır, lakin ML tövsiyə sisteminin hissəsi deyil.",
        )
    )

    st.markdown(t("#### Future work (not implemented)", "#### Gələcək iş (həyata keçirilməyib)"))
    st.markdown(
        t(
            "- **Context-aware choice:** let the use case set the priority, e.g. medical images → "
            "high PSNR, military images → low BER.\n"
            "- **Attack scenarios:** choose the strategy for the attacks a photo is expected to face.\n"
            "- **More embedding methods,** so the recommender has more real alternatives to choose from.\n"
            "- **A unique watermark per user (traceability):** the robust mode already carries a 32-bit "
            "ID; the next step is to give every recipient their own ID automatically and keep a "
            "register, so a leaked copy can be traced back to the person it was given to.\n"
            "- **Camera capture and the recommender:** the robust mode does not yet survive "
            "photographing a screen (perspective), and it is not among the strategies the ML "
            "models choose from; adding it needs new measurements and retraining.",
            "- **Kontekstə görə seçim:** istifadə sahəsi prioriteti təyin etsin, məsələn tibbi "
            "şəkillər → yüksək PSNR, hərbi şəkillər → aşağı BER.\n"
            "- **Hücum ssenariləri:** strategiyanı şəklin qarşılaşacağı hücumlara görə seçmək.\n"
            "- **Daha çox yerləşdirmə metodu,** ki tövsiyə sistemi seçmək üçün daha çox real alternativə "
            "malik olsun.\n"
            "- **Hər istifadəçiyə unikal su nişanı (izlənə bilmə):** dayanıqlı rejim artıq 32-bit ID "
            "daşıyır; növbəti addım hər alıcıya avtomatik öz ID-sini vermək və reyestr saxlamaqdır ki, "
            "sızan nüsxənin kimə verildiyi müəyyən edilə bilsin.\n"
            "- **Kamera ilə çəkiliş və tövsiyə sistemi:** dayanıqlı rejim ekranın kamera ilə "
            "çəkilməsinə (perspektiv) hələ dözmür və ML modellərinin seçdiyi strategiyalar arasında "
            "deyil; onu əlavə etmək yeni ölçmələr və yenidən öyrətmə tələb edir.",
        )
    )
