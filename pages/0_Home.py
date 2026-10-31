"""
AWSRE Framework
Home page: the simple, product-style workflow.

Two actions, like commercial invisible-watermark tools:

    Protect  one or more photos -> AWSRE picks the strategy for EACH photo
             -> download the protected photos
    Verify   suspect photo + original -> watermark found / not found

For every protected photo the page also shows the scientific part:
the predicted outcome of all strategies, why one was chosen, the prediction
checked against a real measurement (16 training attacks), robustness per
attack, and the watermark extracted before and after attacks. Other
strategies are measured only when the user asks to compare with them.

Everything runs through core/protect.py and core/recommender.py, which use
the trained models and the weight-free Pareto + ideal-point rule.
"""

from __future__ import annotations

import io
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.protect import (
    FOUND_BER_THRESHOLD,
    MIN_SIDE,
    TRAINING_ATTACKS,
    default_mark,
    mark_from_image,
    measure_candidate,
    protect,
    verify,
)
from core.recommender import TRAINED_ALPHAS, TRAINED_METHODS
from core.robust_watermark import MAX_ID, protect_robust, real_world_test, verify_blind
from core.robust_watermark import MIN_SIDE as ROBUST_MIN_SIDE
from core.recommender import load_models as _load_models_uncached
from i18n import t
from research_results import show_research_results

try:
    st.set_page_config(page_title="AWSRE", page_icon="🛡️", layout="centered")
except st.errors.StreamlitAPIException:
    pass

TEAL = "#0A6C70"
ORANGE = "#B0501F"
PHOTO_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]


# ============================================================
# HELPERS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    return _load_models_uncached()


def read_rgb(uploaded) -> np.ndarray:
    return np.asarray(Image.open(uploaded).convert("RGB"), dtype=np.uint8)


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def zip_bytes(entries) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            stem = entry["name"].rsplit(".", 1)[0]
            archive.writestr(f"{stem}_awsre.png", png_bytes(entry["result"].protected))
    return buffer.getvalue()


def mark_preview(mark: np.ndarray) -> np.ndarray:
    return (np.kron(mark, np.ones((4, 4), dtype=np.uint8)) * 255).astype(np.uint8)


def attack_label(attack: str, parameter: float) -> str:
    labels = {
        "jpeg": t("JPEG q{p:g}", "JPEG q{p:g}"),
        "gaussian_noise": t("Noise σ{p:g}", "Küy σ{p:g}"),
        "gaussian_blur": t("Blur k{p:g}", "Bulanıqlıq k{p:g}"),
        "crop": t("Crop {p:.0%}", "Kəsmə {p:.0%}"),
        "rotation": t("Rotation {p:g}°", "Fırlatma {p:g}°"),
        "brightness": t("Brightness {p:+g}", "Parlaqlıq {p:+g}"),
        "contrast": t("Contrast {p:+g}", "Kontrast {p:+g}"),
    }
    return labels.get(attack, attack + " {p:g}").format(p=parameter)


def strategy_name(row) -> str:
    return ("AWSRE: " if row.is_awsre else "") + f"{row.method}, α = {row.alpha:g}"


def custom_mark_input(key: str):
    """Optional logo upload. Returns (mark, is_custom)."""
    with st.expander(t("Use your own logo (optional)", "Öz loqonuz (istəyə bağlı)")):
        logo = st.file_uploader(
            t("Logo image", "Loqo şəkli"),
            type=["png", "jpg", "jpeg", "bmp"],
            key=key,
        )
        mark = mark_from_image(np.asarray(Image.open(logo).convert("L"))) if logo else default_mark()
        st.image(
            mark_preview(mark),
            caption=t("What gets hidden (32×32)", "Gizlədilən nişan (32×32)"),
            width=128,
        )
        st.caption(
            t(
                "Bold, simple logos work best. Thin text disappears at 32×32.",
                "Qalın, sadə loqolar ən yaxşısıdır. Nazik yazı 32×32-də itir.",
            )
        )
    return mark, logo is not None


# ============================================================
# DETAIL SECTIONS FOR ONE PROTECTED PHOTO
# ============================================================

def show_metrics(awsre_row) -> None:
    psnr_column, ssim_column, ber_column, corr_column = st.columns(4)
    with psnr_column:
        st.metric("PSNR", f"{awsre_row.psnr:.1f} dB", help=t(
            "Invisibility. Above ~40 dB the change cannot be seen.",
            "Görünməzlik. ~40 dB-dən yuxarı dəyişiklik gözlə görünmür.",
        ))
    with ssim_column:
        st.metric("SSIM", f"{awsre_row.ssim:.3f}", help=t(
            "Structural similarity to the original (1 = identical).",
            "Orijinala struktur oxşarlığı (1 = eyni).",
        ))
    with ber_column:
        st.metric(t("Wrong bits (BER)", "Səhv bitlər (BER)"), f"{awsre_row.mean_attack_ber:.1%}", help=t(
            "Average over the 16 attacks. 0% = perfect, ~50% = lost.",
            "16 hücum üzrə ortalama. 0% = mükəmməl, ~50% = itib.",
        ))
    with corr_column:
        st.metric(t("Correlation", "Korrelyasiya"), f"{awsre_row.mean_attack_correlation:.2f}", help=t(
            "Similarity of the recovered mark to the embedded one, averaged over the 16 attacks (1 = identical).",
            "Çıxarılan nişanın yerləşdirilənə oxşarlığı, 16 hücum üzrə ortalama (1 = eyni).",
        ))
    st.caption(
        t(
            "Measured on this photo: PSNR/SSIM against the original, BER and "
            "correlation after the 16 attacks the models were trained on.",
            "Bu şəkil üzərində ölçülüb: PSNR/SSIM orijinala görə, BER və "
            "korrelyasiya modellərin öyrədildiyi 16 hücumdan sonra.",
        )
    )


def show_extraction(awsre_row, mark: np.ndarray) -> None:
    st.markdown(t("#### Watermark extraction", "#### Su nişanının çıxarılması"))
    embedded_column, clean_column, *attack_columns = st.columns(2 + len(awsre_row.gallery))
    with embedded_column:
        st.image(mark_preview(mark), caption=t("Embedded", "Yerləşdirilən"), use_container_width=True)
    with clean_column:
        st.image(
            mark_preview(awsre_row.clean_extracted_mark),
            caption=t("No attack · {b:.0%} wrong", "Hücumsuz · {b:.0%} səhv").format(b=awsre_row.clean_ber),
            use_container_width=True,
        )
    for column, item in zip(attack_columns, awsre_row.gallery):
        with column:
            st.image(
                mark_preview(item.extracted_mark),
                caption=f"{attack_label(item.attack, item.parameter)} · {item.ber:.0%}",
                use_container_width=True,
            )

    with st.expander(t("See the attacked images", "Hücumlu şəkillərə bax")):
        image_columns = st.columns(len(awsre_row.gallery))
        for column, item in zip(image_columns, awsre_row.gallery):
            with column:
                st.image(
                    item.attacked_thumbnail,
                    caption=attack_label(item.attack, item.parameter),
                    use_container_width=True,
                )
    st.caption(
        t(
            "The mark recovered from the protected photo, without and after attacks. "
            "Rotation and cropping destroy it: a known limit of these methods.",
            "Qorunmuş şəkildən çıxarılan nişan: hücumsuz və hücumlardan sonra. "
            "Fırlatma və kəsmə onu məhv edir: bu metodların məlum məhdudiyyətidir.",
        )
    )


def show_strategy_grid(result) -> None:
    st.markdown(
        t(
            "Before embedding anything, AWSRE predicted how all {n} strategies "
            "would do on this photo from its measured features, and picked the "
            "best balance of invisibility, robustness and speed: **{m}, α = {a:g}**.",
            "AWSRE heç nə yerləşdirmədən əvvəl bu şəklin ölçülmüş "
            "xüsusiyyətlərinə görə {n} strategiyanın nəticəsini proqnozlaşdırdı "
            "və görünməzlik, davamlılıq və sürət arasında ən balanslısını "
            "seçdi: **{m}, α = {a:g}**.",
        ).format(n=len(result.recommendation.candidates), m=result.method, a=result.alpha)
    )

    grid = result.recommendation.candidates[
        ["method", "alpha", "pareto_optimal", "pred_clean_psnr", "pred_mean_attack_ber"]
    ].copy()
    grid["alpha_label"] = grid["alpha"].map(lambda a: f"α {a:g}")
    grid["is_winner"] = (grid["method"] == result.method) & (grid["alpha"] == result.alpha)
    # Each cell shows both sides of the trade-off: invisibility on top,
    # robustness below. Showing PSNR alone made the highest-PSNR cell look
    # like the obvious winner.
    grid["label"] = [
        ("🏆 " if win else ("✓ " if pareto else "")) + f"{psnr:.1f} dB"
        for win, pareto, psnr in zip(grid["is_winner"], grid["pareto_optimal"], grid["pred_clean_psnr"])
    ]
    grid["ber_label"] = grid["pred_mean_attack_ber"].map(lambda ber: t("{b:.1%} wrong", "{b:.1%} səhv").format(b=ber))
    alpha_order = [f"α {a:g}" for a in TRAINED_ALPHAS]
    base = alt.Chart(grid).encode(
        x=alt.X("alpha_label:N", sort=alpha_order, title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("method:N", sort=TRAINED_METHODS, title=None),
    )
    cells = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "pred_clean_psnr:Q",
            legend=None,
            scale=alt.Scale(scheme=alt.SchemeParams(name="tealblues", extent=[0.05, 0.5]), zero=False),
        ),
        tooltip=[
            alt.Tooltip("method:N", title=t("Method", "Metod")),
            alt.Tooltip("alpha:Q", title="α"),
            alt.Tooltip("pred_clean_psnr:Q", title=t("Predicted PSNR (dB)", "Proqnoz PSNR (dB)"), format=".2f"),
            alt.Tooltip("pred_mean_attack_ber:Q", title=t("Predicted wrong bits", "Proqnoz səhv bitlər"), format=".1%"),
        ],
    )
    winner_box = (
        alt.Chart(grid[grid["is_winner"]])
        .mark_rect(fill=None, stroke=TEAL, strokeWidth=4)
        .encode(x=alt.X("alpha_label:N", sort=alpha_order), y=alt.Y("method:N", sort=TRAINED_METHODS))
    )
    text = base.mark_text(fontSize=13, fontWeight="bold", dy=-8).encode(text="label:N", color=alt.value("#13201F"))
    ber_text = base.mark_text(fontSize=11, dy=9).encode(text="ber_label:N", color=alt.value("#7A2E0E"))
    st.altair_chart(
        (cells + winner_box + text + ber_text).properties(height=64 * len(TRAINED_METHODS) + 36),
        use_container_width=True,
    )
    st.caption(
        t(
            "Each cell: predicted PSNR (dB, higher = more invisible) and predicted "
            "wrong bits after attacks (lower = more robust). The highest PSNR is "
            "not automatically best: it usually comes with more wrong bits. ✓ = "
            "Pareto-optimal: no other strategy is "
            "predicted to be at least as good on all three objectives (distortion, "
            "wrong bits after attacks, time) and better on one. 🏆 = among the ✓, "
            "the one closest to the ideal point: AWSRE's choice.",
            "Hər hüceyrə: proqnozlaşdırılan PSNR (dB, yüksək = daha görünməz) və "
            "hücumdan sonra proqnozlaşdırılan səhv bitlər (aşağı = daha davamlı). "
            "Ən yüksək PSNR avtomatik ən yaxşı deyil: adətən daha çox səhv bitlə "
            "gəlir. ✓ = Pareto-optimal: heç bir "
            "başqa strategiyanın üç meyarın (təhrif, hücumdan sonra səhv bitlər, "
            "vaxt) hamısında ən azı onun qədər yaxşı və birində daha yaxşı olacağı "
            "proqnozlaşdırılmır. 🏆 = ✓-lar arasında ideal nöqtəyə ən yaxın olan: "
            "AWSRE-nin seçimi.",
        )
    )


def show_measurement_table(comparison) -> None:
    # One row per metric; for every strategy a "predicted" and a "real" column.
    metrics = [
        ("PSNR ↑", "predicted_psnr", "psnr", "{:.2f} dB"),
        ("SSIM ↑", "predicted_ssim", "ssim", "{:.4f}"),
        ("MSE ↓", "predicted_mse", "mse", "{:.2f}"),
        (t("BER (wrong bits) ↓", "BER (səhv bitlər) ↓"), "predicted_ber", "mean_attack_ber", "{:.1%}"),
        (t("NC (correlation) ↑", "NC (korrelyasiya) ↑"), "predicted_correlation", "mean_attack_correlation", "{:.3f}"),
    ]

    def fmt(template: str, value: float) -> str:
        return "—" if value != value else template.format(value)

    columns = {t("Metric", "Göstərici"): [label for label, *_ in metrics]}
    for row in comparison:
        name = ("🏆 " if row.is_awsre else "") + f"{row.method} α{row.alpha:g}"
        columns[f"{name} · " + t("predicted", "proqnoz")] = [fmt(f, getattr(row, p)) for _, p, _, f in metrics]
        columns[f"{name} · real"] = [fmt(f, getattr(row, r)) for _, _, r, f in metrics]
    st.dataframe(pd.DataFrame(columns), hide_index=True, use_container_width=True)

    awsre_position = next(i for i, row in enumerate(comparison) if row.is_awsre)
    awsre_row = comparison[awsre_position]
    others = [row for row in comparison if not row.is_awsre]
    caption = t(
        "Predicted before anything was embedded: PSNR and SSIM by the clean-quality "
        "model, MSE and BER by the strategy model, NC by the attack model (all "
        "averaged over the 16 training attacks where attacks apply). Real: measured "
        "now on this photo. Prediction error for AWSRE's choice: {pe:.2f} dB PSNR, "
        "{be:.1%} BER. ",
        "Heç nə yerləşdirilmədən əvvəl proqnozlaşdırılıb: PSNR və SSIM təmiz-keyfiyyət "
        "modeli, MSE və BER strategiya modeli, NC hücum modeli ilə (hücumla bağlı "
        "olanlar 16 təlim hücumu üzrə ortalamadır). Real: bu şəkil üzərində indi "
        "ölçülüb. AWSRE seçimi üçün proqnoz xətası: PSNR {pe:.2f} dB, BER {be:.1%}. ",
    ).format(
        pe=abs(awsre_row.predicted_psnr - awsre_row.psnr),
        be=abs(awsre_row.predicted_ber - awsre_row.mean_attack_ber),
    )
    if not others:
        caption += t(
            "The other strategies were not embedded: AWSRE chose from their "
            "predictions. Compare with any of them below.",
            "Digər strategiyalar yerləşdirilmədi: AWSRE onların proqnozuna görə "
            "seçdi. İstənilən biri ilə aşağıda müqayisə edə bilərsiniz.",
        )
    else:
        beaten = any(
            row.psnr >= awsre_row.psnr
            and row.mean_attack_ber <= awsre_row.mean_attack_ber
            and (row.psnr > awsre_row.psnr or row.mean_attack_ber < awsre_row.mean_attack_ber)
            for row in others
        )
        caption += (
            t(
                "A compared strategy was better on both real PSNR and real BER here: "
                "AWSRE decides from predictions and lands on a best trade-off in "
                "94.7% of the study's test cases, not all of them.",
                "Burada müqayisə olunan strategiya həm real PSNR, həm də real BER-də "
                "daha yaxşı çıxdı: AWSRE proqnoza əsasən qərar verir və tədqiqatın "
                "test hallarının hamısında yox, 94,7%-də ən yaxşı kompromisə düşür.",
            )
            if beaten
            else t(
                "No compared strategy beats AWSRE's choice on both PSNR and wrong bits.",
                "Müqayisə olunan heç bir strategiya AWSRE seçimini həm PSNR, həm "
                "də səhv bitlərdə üstələmir.",
            )
        )
    st.caption(caption)


def show_robustness_chart(comparison) -> None:
    order = [attack_label(a, p) for a, p in TRAINING_ATTACKS]
    data = pd.DataFrame(
        [
            {
                "attack": attack_label(outcome.attack, outcome.parameter),
                "strategy": strategy_name(row),
                "ber": outcome.ber,
                "predicted_ber": outcome.predicted_ber,
            }
            for row in comparison
            for outcome in row.attacks
        ]
    )
    strategies = [strategy_name(row) for row in comparison]
    palette = [TEAL, ORANGE, "#6B5CA5", "#8A8F2E", "#A23B72"]
    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X("attack:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45)),
        xOffset=alt.XOffset("strategy:N", sort=strategies),
        y=alt.Y(
            "ber:Q",
            title=t("Wrong bits (lower is better)", "Səhv bitlər (aşağı daha yaxşı)"),
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domain=[0, 0.6]),
        ),
        color=alt.Color(
            "strategy:N",
            sort=strategies,
            scale=alt.Scale(domain=strategies, range=palette[: len(strategies)]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=[
            alt.Tooltip("strategy:N", title=t("Strategy", "Strategiya")),
            alt.Tooltip("attack:N", title=t("Attack", "Hücum")),
            alt.Tooltip("ber:Q", title=t("Wrong bits", "Səhv bitlər"), format=".1%"),
        ],
    )
    predicted_marks = alt.Chart(data.dropna(subset=["predicted_ber"])).mark_tick(
        color="#13201F", thickness=2, size=10
    ).encode(
        x=alt.X("attack:N", sort=order),
        xOffset=alt.XOffset("strategy:N", sort=strategies),
        y="predicted_ber:Q",
        tooltip=[
            alt.Tooltip("strategy:N", title=t("Strategy", "Strategiya")),
            alt.Tooltip("attack:N", title=t("Attack", "Hücum")),
            alt.Tooltip("predicted_ber:Q", title=t("Predicted wrong bits", "Proqnoz səhv bitlər"), format=".1%"),
        ],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [FOUND_BER_THRESHOLD]})).mark_rule(
        strokeDash=[5, 4], color="#888888"
    ).encode(y="y:Q")
    st.altair_chart((bars + predicted_marks + threshold).properties(height=300), use_container_width=True)
    st.caption(
        t(
            "Bars: measured (each strategy embedded, attacked and extracted). Black "
            "ticks: the attack model's prediction for the same attack, made before "
            "embedding. Above the dashed line ({th:.0%}) the watermark counts as lost.",
            "Sütunlar: ölçülüb (hər strategiya yerləşdirilib, hücuma məruz qalıb və "
            "çıxarılıb). Qara xətlər: həmin hücum üçün hücum modelinin yerləşdirmədən "
            "əvvəlki proqnozu. Qırıq xəttdən ({th:.0%}) yuxarıda su nişanı itmiş sayılır.",
        ).format(th=FOUND_BER_THRESHOLD)
    )


def show_compare_controls(entry_index: int, entry) -> None:
    st.markdown(t("**Compare with another strategy**", "**Başqa strategiya ilə müqayisə et**"))
    method_column, alpha_column, button_column = st.columns([1.2, 1, 1])
    with method_column:
        other_method = st.selectbox(t("Method", "Metod"), TRAINED_METHODS, key="home_compare_method")
    with alpha_column:
        other_alpha = st.selectbox("α", TRAINED_ALPHAS, format_func=lambda a: f"{a:g}", key="home_compare_alpha")
    with button_column:
        st.write("")
        clicked = st.button(t("Compare", "Müqayisə et"), use_container_width=True, key="home_compare_button")

    if not clicked:
        return
    if any(row.method == other_method and row.alpha == float(other_alpha) for row in entry["comparison"]):
        st.info(t("That strategy is already in the table.", "Bu strategiya artıq cədvəldədir."))
        return
    with st.spinner(t(
        "Embedding {m}, α = {a:g} and running 16 attacks...",
        "{m}, α = {a:g} yerləşdirilir və 16 hücum tətbiq olunur...",
    ).format(m=other_method, a=other_alpha)):
        measured = measure_candidate(
            entry["original"], entry["mark"], entry["result"].recommendation, other_method, float(other_alpha),
            attack_model=attack_model,
        )
    update_entry(entry_index, comparison=entry["comparison"] + [measured])
    st.rerun()


def update_entry(index: int, **changes) -> None:
    batch = dict(st.session_state["home_protect_batch"])
    entries = list(batch["entries"])
    entries[index] = {**entries[index], **changes}
    batch["entries"] = entries
    st.session_state["home_protect_batch"] = batch


def show_photo_details(index: int, entry) -> None:
    result = entry["result"]

    if entry["comparison"] is None:
        with st.spinner(t(
            "Checking the prediction against 16 attacks...",
            "Proqnoz 16 hücumla yoxlanılır...",
        )):
            awsre_row = measure_candidate(
                entry["original"], entry["mark"], result.recommendation, result.method, result.alpha,
                attack_model=attack_model,
            )
        update_entry(index, comparison=[awsre_row])
        entry = st.session_state["home_protect_batch"]["entries"][index]

    comparison = entry["comparison"]
    awsre_row = next(row for row in comparison if row.is_awsre)

    st.success(
        t(
            "Protected with {m}, α = {a:g}. The difference is invisible to the eye.",
            "{m}, α = {a:g} ilə qorundu. Fərq gözlə görünmür.",
        ).format(m=result.method, a=result.alpha)
    )

    before, after = st.columns(2)
    with before:
        st.image(entry["original"], caption=t("Original", "Orijinal"), use_container_width=True)
    with after:
        st.image(result.protected, caption=t("Protected", "Qorunmuş"), use_container_width=True)

    stem = entry["name"].rsplit(".", 1)[0]
    st.download_button(
        t("Download protected photo (PNG)", "Qorunmuş şəkli endir (PNG)"),
        data=png_bytes(result.protected),
        file_name=f"{stem}_awsre.png",
        mime="image/png",
        type="primary",
        use_container_width=True,
        key=f"home_protect_download_{index}",
    )
    st.caption(t("Keep the original: verification compares against it.", "Orijinalı saxlayın: yoxlama onunla müqayisə edir."))

    show_metrics(awsre_row)

    st.divider()
    st.subheader(t("Why this strategy?", "Niyə bu strategiya?"))
    show_strategy_grid(result)
    st.markdown(t("**Prediction vs reality**", "**Proqnoz vs reallıq**"))
    show_measurement_table(comparison)
    show_compare_controls(index, entry)

    st.divider()
    st.subheader(t("Robustness to attacks", "Hücumlara davamlılıq"))
    show_robustness_chart(comparison)
    show_extraction(awsre_row, entry["mark"])

    st.divider()
    st.subheader(t("Real-world attacks: AWSRE vs robust method", "Real həyat hücumları: AWSRE vs dayanıqlı metod"))
    show_robust_comparison(index, entry)



# ============================================================
# ROBUST BLIND MODE (not one of the ML recommender's strategies)
# ============================================================

def owner_id_input(key: str) -> int:
    return int(st.number_input(
        t("ID hidden in the photo (owner or recipient number)", "Şəklə gizlədilən ID (sahib və ya alıcı nömrəsi)"),
        min_value=0, max_value=MAX_ID, value=1, step=1, key=key,
        help=t(
            "A 32-bit number. Verification reads it back from a copy without the original.",
            "32 bitlik nömrə. Yoxlama onu orijinal olmadan nüsxədən geri oxuyur.",
        ),
    ))


def show_real_world_table(rows, with_awsre: bool) -> None:
    def yes_no(found):
        return "✅ " + t("found", "tapıldı") if found else "❌ " + t("lost", "itdi")

    table = []
    for row in rows:
        line = {t("Attack", "Hücum"): t(row.label_en, row.label_az)}
        if with_awsre:
            line[t("AWSRE choice (with original)", "AWSRE seçimi (orijinalla)")] = (
                f"{yes_no(row.awsre_found)} · {row.awsre_ber:.0%} " + t("wrong", "səhv")
            )
        line[t("Robust (no original)", "Dayanıqlı (orijinalsız)")] = (
            yes_no(row.robust_found) + (f" · ID {row.robust_id}" if row.robust_found else "")
        )
        table.append(line)
    st.dataframe(table, hide_index=True, use_container_width=True)
    robust_count = sum(r.robust_found for r in rows)
    if with_awsre:
        awsre_count = sum(bool(r.awsre_found) for r in rows)
        st.caption(t(
            "Measured now on this photo, {n} attacks. AWSRE's choice survived {a}, the robust method {r}. "
            "AWSRE's photo is checked the site's way (original given, 1,024 bits, found at ≤ 30% wrong); "
            "the robust photo with the key only, and counts as found only if the exact ID comes back.",
            "Bu şəkil üzərində indi ölçülüb, {n} hücum. AWSRE seçimi {a}-ə, dayanıqlı metod {r}-ə dözdü. "
            "AWSRE şəkli saytın qaydası ilə yoxlanılır (orijinal verilir, 1 024 bit, ≤ 30% səhvdə tapılmış sayılır); "
            "dayanıqlı şəkil yalnız açarla yoxlanılır və ID tam düzgün oxunanda tapılmış sayılır.",
        ).format(n=len(rows), a=awsre_count, r=robust_count))
    else:
        st.caption(t(
            "Measured now on this photo: each attack applied, then checked with the key only "
            "(no original). Found = the exact ID came back. Survived {r} of {n}.",
            "Bu şəkil üzərində indi ölçülüb: hər hücum tətbiq olunub, sonra yalnız açarla "
            "(orijinalsız) yoxlanılıb. Tapıldı = ID tam düzgün oxundu. {n} hücumdan {r}-ə dözdü.",
        ).format(n=len(rows), r=robust_count))


def show_robust_comparison(index: int, entry) -> None:
    """AWSRE's choice next to the robust blind method, same photo, same attacks."""
    st.markdown(t(
        "The 3 methods AWSRE chooses from need the original to verify and do not survive "
        "cropping. The separate **robust method** (not one of the 15 strategies, no "
        "predictions) is verified without the original. Compare both on real-world attacks:",
        "AWSRE-nin seçdiyi 3 metod yoxlama üçün orijinal tələb edir və kəsməyə dözmür. Ayrıca "
        "**dayanıqlı metod** (15 strategiyadan biri deyil, proqnozu yoxdur) orijinalsız yoxlanılır. "
        "Hər ikisini real həyat hücumlarında müqayisə edin:",
    ))
    if entry.get("real_world") is None:
        if not st.button(
            t("Compare with the robust method", "Dayanıqlı metodla müqayisə et"),
            use_container_width=True, key=f"home_robust_compare_{index}",
        ):
            return
        with st.spinner(t("Embedding the robust watermark and running the attacks...",
                          "Dayanıqlı nişan yerləşdirilir və hücumlar tətbiq olunur...")):
            robust = protect_robust(entry["original"], 1)
            rows = real_world_test(robust.protected, 1, original=entry["original"],
                                   awsre_protected=entry["result"].protected, mark=entry["mark"])
        update_entry(index, real_world={"robust": robust, "rows": rows})
        entry = st.session_state["home_protect_batch"]["entries"][index]

    robust, rows = entry["real_world"]["robust"], entry["real_world"]["rows"]
    awsre_row = next(row for row in entry["comparison"] if row.is_awsre)
    st.dataframe(
        [
            {t("Method", "Metod"): f"AWSRE: {awsre_row.method}, α = {awsre_row.alpha:g}",
             "PSNR": f"{awsre_row.psnr:.1f} dB", "SSIM": f"{awsre_row.ssim:.4f}",
             t("Needs the original to verify", "Yoxlama üçün orijinal lazımdır"): t("Yes", "Bəli")},
            {t("Method", "Metod"): t("Robust (tiled, 32-bit ID)", "Dayanıqlı (kafel, 32-bit ID)"),
             "PSNR": f"{robust.psnr:.1f} dB", "SSIM": f"{robust.ssim:.4f}",
             t("Needs the original to verify", "Yoxlama üçün orijinal lazımdır"): t("No", "Xeyr")},
        ],
        hide_index=True, use_container_width=True,
    )
    show_real_world_table(rows, with_awsre=True)
    st.caption(t(
        "Trade-off: the robust mark is stronger, so its PSNR is usually lower. It does not yet "
        "survive photographing a screen with a camera.",
        "Güzəşt: dayanıqlı nişan daha güclüdür, ona görə PSNR adətən aşağıdır. Ekranın kamera "
        "ilə çəkilməsinə hələ dözmür.",
    ))


def show_robust_details(index: int, entry) -> None:
    result = entry["result"]
    st.success(t(
        "Protected with the robust method. ID {i} is hidden in the photo.",
        "Dayanıqlı metodla qorundu. Şəklə {i} ID-si gizlədildi.",
    ).format(i=result.owner_id))
    before, after = st.columns(2)
    with before:
        st.image(entry["original"], caption=t("Original", "Orijinal"), use_container_width=True)
    with after:
        st.image(result.protected, caption=t("Protected", "Qorunmuş"), use_container_width=True)
    stem = entry["name"].rsplit(".", 1)[0]
    st.download_button(
        t("Download protected photo (PNG)", "Qorunmuş şəkli endir (PNG)"),
        data=png_bytes(result.protected), file_name=f"{stem}_awsre_robust.png", mime="image/png",
        type="primary", use_container_width=True, key=f"home_robust_download_{index}",
    )
    st.caption(t("Verification does not need the original.", "Yoxlama üçün orijinal lazım deyil."))
    psnr_column, ssim_column = st.columns(2)
    with psnr_column:
        st.metric("PSNR", f"{result.psnr:.1f} dB")
    with ssim_column:
        st.metric("SSIM", f"{result.ssim:.3f}")

    st.divider()
    st.subheader(t("Robustness to real-world attacks", "Real həyat hücumlarına davamlılıq"))
    if entry.get("rows") is None:
        if not st.button(t("Run the attacks", "Hücumları tətbiq et"), use_container_width=True,
                         key=f"home_robust_attacks_{index}"):
            return
        with st.spinner(t("Running the attacks...", "Hücumlar tətbiq olunur...")):
            rows = real_world_test(result.protected, result.owner_id)
        batch = dict(st.session_state["home_robust_batch"])
        entries = list(batch["entries"])
        entries[index] = {**entries[index], "rows": rows}
        batch["entries"] = entries
        st.session_state["home_robust_batch"] = batch
        entry = entries[index]
    show_real_world_table(entry["rows"], with_awsre=False)


# ============================================================
# HEADER
# ============================================================

st.title("AWSRE")
st.markdown(
    t(
        "Add an invisible watermark to your photos, and check later whether a copy carries it.",
        "Şəkillərinizə görünməz su nişanı əlavə edin və sonra hər hansı nüsxədə onun olub-olmadığını yoxlayın.",
    )
)

clean_model, strategy_model, attack_model, _attack_path, model_errors = load_models()
if model_errors:
    st.error(
        t(
            "The trained models were not found. Run experiments/train_performance_models.py first.",
            "Öyrədilmiş modellər tapılmadı. Əvvəlcə experiments/train_performance_models.py-ı işə salın.",
        )
    )
    st.stop()

protect_tab, verify_tab, results_tab = st.tabs(
    [t("Protect", "Qoru"), t("Verify", "Yoxla"), t("Research results", "Elmi nəticələr")]
)

with results_tab:
    show_research_results()


# ============================================================
# PROTECT
# ============================================================

with protect_tab:
    protect_mode = st.radio(
        t("Protection type", "Qoruma növü"),
        ["awsre", "robust"],
        format_func=lambda m: {
            "awsre": t("AWSRE: ML picks the strategy (verify with the original)",
                       "AWSRE: strategiyanı ML seçir (orijinalla yoxlama)"),
            "robust": t("Robust: survives crop, JPEG, downscale (verify without the original)",
                        "Dayanıqlı: kəsmə, JPEG, kiçiltməyə dözür (orijinalsız yoxlama)"),
        }[m],
        key="home_protect_mode",
    )
    if protect_mode == "awsre":
        photo_files = st.file_uploader(
            t("Photos to protect (one or more)", "Qorunacaq şəkillər (bir və ya bir neçə)"),
            type=PHOTO_TYPES,
            accept_multiple_files=True,
            key="home_protect_photos",
        )
        protect_mark, _ = custom_mark_input("home_protect_logo")

        if st.button(
            t("Protect", "Qoru"),
            type="primary",
            use_container_width=True,
            disabled=not photo_files,
            key="home_protect_button",
        ):
            entries, skipped = [], []
            progress = st.progress(0.0, text=t("Choosing a strategy for each photo...", "Hər şəkil üçün strategiya seçilir..."))
            for position, photo_file in enumerate(photo_files):
                photo = read_rgb(photo_file)
                try:
                    result = protect(photo, protect_mark, clean_model, strategy_model)
                except ValueError:
                    skipped.append(photo_file.name)
                else:
                    entries.append({
                        "name": photo_file.name,
                        "original": photo,
                        "mark": protect_mark,
                        "result": result,
                        "comparison": None,
                        "real_world": None,
                    })
                progress.progress((position + 1) / len(photo_files))
            progress.empty()
            st.session_state["home_protect_batch"] = {
                "names": tuple(f.name for f in photo_files),
                "entries": entries,
                "skipped": skipped,
            }

        batch = st.session_state.get("home_protect_batch")
        current_names = tuple(f.name for f in photo_files) if photo_files else ()
        if batch is not None and batch["names"] == current_names:
            if batch["skipped"]:
                st.warning(
                    t(
                        "Skipped (smaller than {n} px): {names}",
                        "Ötürüldü ({n} px-dən kiçik): {names}",
                    ).format(n=MIN_SIDE, names=", ".join(batch["skipped"]))
                )

            entries = batch["entries"]
            if len(entries) > 1:
                st.markdown(t("#### Each photo got its own strategy", "#### Hər şəkil öz strategiyasını aldı"))
                st.dataframe(
                    [
                        {
                            t("Photo", "Şəkil"): entry["name"],
                            t("Size", "Ölçü"): f"{entry['original'].shape[1]}×{entry['original'].shape[0]}",
                            t("AWSRE choice", "AWSRE seçimi"): f"{entry['result'].method}, α = {entry['result'].alpha:g}",
                            t("PSNR predicted", "PSNR proqnoz"): f"{entry['result'].recommendation.winner['pred_clean_psnr']:.1f} dB",
                            t("Wrong bits predicted", "Səhv bitlər proqnoz"): f"{entry['result'].recommendation.winner['pred_mean_attack_ber']:.1%}",
                        }
                        for entry in entries
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                distinct = len({(e["result"].method, e["result"].alpha) for e in entries})
                st.caption(
                    t(
                        "{k} different strategies for {n} photos. Each photo was analysed "
                        "separately from its own features.",
                        "{n} şəkil üçün {k} fərqli strategiya. Hər şəkil öz xüsusiyyətlərinə "
                        "görə ayrıca analiz edildi.",
                    ).format(k=distinct, n=len(entries))
                )
                st.download_button(
                    t("Download all protected photos (ZIP)", "Bütün qorunmuş şəkilləri endir (ZIP)"),
                    data=zip_bytes(entries),
                    file_name="awsre_protected.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="home_protect_zip",
                )
                selected = st.selectbox(
                    t("Show details for", "Ətraflı bax"),
                    range(len(entries)),
                    format_func=lambda i: entries[i]["name"],
                    key="home_protect_selected",
                )
                st.divider()
            else:
                selected = 0

            if entries:
                show_photo_details(selected, entries[selected])
    else:
        robust_files = st.file_uploader(
            t("Photos to protect (one or more)", "Qorunacaq şəkillər (bir və ya bir neçə)"),
            type=PHOTO_TYPES, accept_multiple_files=True, key="home_robust_photos",
        )
        robust_id = owner_id_input("home_robust_id")
        if st.button(t("Protect", "Qoru"), type="primary", use_container_width=True,
                     disabled=not robust_files, key="home_robust_button"):
            entries, skipped = [], []
            progress = st.progress(0.0)
            for position, photo_file in enumerate(robust_files):
                photo = read_rgb(photo_file)
                try:
                    result = protect_robust(photo, robust_id)
                except ValueError:
                    skipped.append(photo_file.name)
                else:
                    entries.append({"name": photo_file.name, "original": photo, "result": result, "rows": None})
                progress.progress((position + 1) / len(robust_files))
            progress.empty()
            st.session_state["home_robust_batch"] = {
                "names": tuple(f.name for f in robust_files), "entries": entries, "skipped": skipped,
            }

        robust_batch = st.session_state.get("home_robust_batch")
        robust_names = tuple(f.name for f in robust_files) if robust_files else ()
        if robust_batch is not None and robust_batch["names"] == robust_names:
            if robust_batch["skipped"]:
                st.warning(t("Skipped (smaller than {n} px): {names}", "Ötürüldü ({n} px-dən kiçik): {names}").format(
                    n=ROBUST_MIN_SIDE, names=", ".join(robust_batch["skipped"])))
            robust_entries = robust_batch["entries"]
            if len(robust_entries) > 1:
                st.download_button(
                    t("Download all protected photos (ZIP)", "Bütün qorunmuş şəkilləri endir (ZIP)"),
                    data=zip_bytes(robust_entries), file_name="awsre_robust.zip", mime="application/zip",
                    use_container_width=True, key="home_robust_zip",
                )
                chosen = st.selectbox(t("Show details for", "Ətraflı bax"), range(len(robust_entries)),
                                      format_func=lambda i: robust_entries[i]["name"], key="home_robust_selected")
                st.divider()
            else:
                chosen = 0
            if robust_entries:
                show_robust_details(chosen, robust_entries[chosen])


# ============================================================
# VERIFY
# ============================================================

with verify_tab:
    suspect_column, original_column = st.columns(2)
    with suspect_column:
        suspect_file = st.file_uploader(
            t("Photo to check", "Yoxlanılacaq şəkil"),
            type=PHOTO_TYPES,
            key="home_verify_suspect",
        )
    with original_column:
        original_file = st.file_uploader(
            t("Your original photo (optional)", "Orijinal şəkliniz (istəyə bağlı)"),
            type=PHOTO_TYPES,
            key="home_verify_original",
        )
    verify_mark, _ = custom_mark_input("home_verify_logo")

    if st.button(
        t("Verify", "Yoxla"),
        type="primary",
        use_container_width=True,
        disabled=suspect_file is None,
        key="home_verify_button",
    ):
        with st.spinner(t("Looking for the watermark...", "Su nişanı axtarılır...")):
            suspect_rgb = read_rgb(suspect_file)
            # 1) robust watermark: key only, no original needed
            blind = verify_blind(suspect_rgb)
            # 2) AWSRE's methods: only possible when the original is given
            outcome = None
            if not blind.found and original_file is not None:
                outcome = verify(suspect_rgb, read_rgb(original_file), verify_mark)
        st.session_state["home_verify_result"] = {
            "names": (suspect_file.name, original_file.name if original_file else None),
            "blind": blind,
            "outcome": outcome,
            "mark": verify_mark,
        }

    saved = st.session_state.get("home_verify_result")
    current_names = (
        suspect_file.name if suspect_file else None,
        original_file.name if original_file else None,
    )
    if saved is not None and saved["names"] == current_names and saved["blind"].found:
        blind = saved["blind"]
        st.success(t("Robust watermark found, no original needed", "Dayanıqlı su nişanı tapıldı, orijinal lazım olmadı"))
        id_column, score_column = st.columns(2)
        with id_column:
            st.metric(t("ID in the photo", "Şəkildəki ID"), f"{blind.user_id}")
        with score_column:
            st.metric(t("Sync score", "Sinxronizasiya göstəricisi"), f"{blind.sync_score:.1f}", help=t(
                "Strength of the hidden pattern's correlation peak in standard deviations. "
                "Accepted at 7 or more, and only if the ID's 16-bit checksum also matches.",
                "Gizli nümunənin korrelyasiya zirvəsinin gücü (standart sapma ilə). 7 və yuxarı "
                "olduqda və ID-nin 16 bitlik yoxlama cəmi də uyğun gəldikdə qəbul olunur.",
            ))
        with st.expander(t("How this was decided", "Bu necə müəyyən edildi")):
            st.markdown(t(
                "The photo's autocorrelation reveals the repeating hidden tile, which gives back the "
                "scale and rotation the copy went through. The photo is then folded into one tile, "
                "aligned with the key's sync pattern, and 48 bits are read (32-bit ID + 16-bit "
                "checksum). On unmarked photos and with wrong keys this test gave 0 false finds in "
                "80 trials (research/robust_tiled).",
                "Şəklin avtokorrelyasiyası təkrarlanan gizli kafeli göstərir, bu da nüsxənin keçdiyi "
                "miqyas və fırlanmanı geri verir. Sonra şəkil bir kafelə qatlanır, açarın "
                "sinxronizasiya nümunəsi ilə hizalanır və 48 bit oxunur (32-bit ID + 16-bit yoxlama "
                "cəmi). Nişansız şəkillərdə və səhv açarla 80 sınaqda 0 yalançı tapıntı oldu "
                "(research/robust_tiled).",
            ))
    elif saved is not None and saved["names"] == current_names and saved["outcome"] is None:
        st.error(t("No watermark found", "Su nişanı tapılmadı"))
        st.caption(t(
            "No robust watermark is in this photo. If it was protected in AWSRE mode, also upload "
            "the original: those methods can only be checked against it.",
            "Bu şəkildə dayanıqlı su nişanı yoxdur. Əgər AWSRE rejimində qorunubsa, orijinalı da "
            "yükləyin: o metodlar yalnız orijinalla yoxlana bilir.",
        ))
    elif saved is not None and saved["names"] == current_names:
        outcome = saved["outcome"]

        if outcome.identical:
            st.info(
                t(
                    "This is the original photo itself, so there is no watermark to find.",
                    "Bu, orijinal şəklin özüdür, ona görə tapılacaq su nişanı yoxdur.",
                )
            )
        elif outcome.found:
            st.success(t("Watermark found", "Su nişanı tapıldı"))
            match_column, expected_column, found_column = st.columns([1.2, 1, 1])
            with match_column:
                st.metric(t("Match", "Uyğunluq"), f"{outcome.match_percent:.0f}%")
            with expected_column:
                st.image(mark_preview(saved["mark"]), caption=t("Your mark", "Sizin nişan"), width=128)
            with found_column:
                st.image(mark_preview(outcome.extracted_mark), caption=t("Found in the photo", "Şəkildə tapılan"), width=128)
        else:
            st.error(t("No watermark found", "Su nişanı tapılmadı"))
            st.caption(
                t(
                    "No robust watermark either. Either this copy was never protected "
                    "with this original, or it was cropped or rotated: AWSRE's methods "
                    "do not survive cropping or rotation (the robust mode does).",
                    "Dayanıqlı su nişanı da yoxdur. Ya bu nüsxə həmin orijinalla "
                    "qorunmayıb, ya da kəsilib və ya fırladılıb: AWSRE metodları kəsmə "
                    "və fırlatmaya davamlı deyil (dayanıqlı rejim isə davamlıdır).",
                )
            )

        with st.expander(t("How this was decided", "Bu necə müəyyən edildi")):
            st.markdown(
                t(
                    "Each trained method's extractor compares the copy with your "
                    "original and reads back 1,024 bits. With no watermark about half "
                    "the bits are wrong (chance). The watermark counts as found when "
                    "at most {th:.0%} are wrong, far below what chance can produce.",
                    "Hər öyrədilmiş metodun çıxarıcısı nüsxəni orijinalınızla müqayisə "
                    "edib 1 024 bit oxuyur. Su nişanı yoxdursa, bitlərin təxminən "
                    "yarısı səhv olur (təsadüf). Səhv bitlər ən çox {th:.0%} olduqda "
                    "su nişanı tapılmış sayılır, bu isə təsadüfən alına bilməz.",
                ).format(th=FOUND_BER_THRESHOLD)
            )
            if outcome.per_method_ber:
                st.dataframe(
                    [
                        {t("Method", "Metod"): method, t("Wrong bits", "Səhv bitlər"): f"{ber:.1%}"}
                        for method, ber in outcome.per_method_ber.items()
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
