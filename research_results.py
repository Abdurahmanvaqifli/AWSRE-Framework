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


def show_research_results() -> None:
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
            "- **Verification needs the original photo** (non-blind DCT), and rotation or cropping "
            "destroys the mark.",
            "- **Bir strategiya üstünlük təşkil edir.** Vaxt meyarı olmadan (yalnız MSE və BER) AWSRE "
            "320 test kontekstindən 318-də DCT α = 15 seçir. Proqnoz əvəzinə real ölçmələrdən istifadə "
            "edən oracle da onların 84%-də DCT α = 15 seçir, yəni bu, modelin meyli deyil, verilənlərin "
            "özünü əks etdirir. Adaptivlik əsasən bulanıqlıqda və minimum PSNR tələb olunanda fayda verir.\n"
            "- **Yoxlama üçün orijinal şəkil lazımdır** (kor olmayan DCT), fırlatma və kəsmə isə "
            "nişanı məhv edir.",
        )
    )

    st.markdown(t("#### Future work (not implemented)", "#### Gələcək iş (həyata keçirilməyib)"))
    st.markdown(
        t(
            "- **Context-aware choice:** let the use case set the priority, e.g. medical images → "
            "high PSNR, military images → low BER.\n"
            "- **Attack scenarios:** choose the strategy for the attacks a photo is expected to face.\n"
            "- **More embedding methods,** so the recommender has more real alternatives to choose from.\n"
            "- **A unique watermark per user (traceability):** instead of the same logo for everyone, "
            "embed the recipient's ID (user metadata) in each copy, so a leaked copy can be traced back "
            "to the person it was given to.\n"
            "- **Robustness to geometric changes:** a blind, self-synchronising watermark that survives "
            "cropping, downscaling, re-encoding and screen or camera capture.",
            "- **Kontekstə görə seçim:** istifadə sahəsi prioriteti təyin etsin, məsələn tibbi "
            "şəkillər → yüksək PSNR, hərbi şəkillər → aşağı BER.\n"
            "- **Hücum ssenariləri:** strategiyanı şəklin qarşılaşacağı hücumlara görə seçmək.\n"
            "- **Daha çox yerləşdirmə metodu,** ki tövsiyə sistemi seçmək üçün daha çox real alternativə "
            "malik olsun.\n"
            "- **Hər istifadəçiyə unikal su nişanı (izlənə bilmə):** hamı üçün eyni loqo əvəzinə hər "
            "nüsxəyə alıcının ID-si (istifadəçi metadatası) yerləşdirilsin ki, sızan nüsxənin kimə "
            "verildiyi müəyyən edilə bilsin.\n"
            "- **Həndəsi dəyişikliklərə davamlılıq:** kəsməyə, kiçiltməyə, yenidən kodlaşdırmaya və "
            "ekran və ya kamera ilə çəkilişə dözən, orijinalsız (kor) və özü sinxronlaşan su nişanı.",
        )
    )
