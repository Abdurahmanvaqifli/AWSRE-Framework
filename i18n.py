"""
AWSRE Framework
Minimal bilingual (Azerbaijani / English) UI helper.

Design: instead of a separate key -> {"az":..., "en":...} registry that has
to be kept in sync by hand across 4 files, every call site passes BOTH
strings inline:

    st.title(t("Comparison", "Müqayisə"))

`t(en, az)` just returns whichever one matches the language currently
selected in st.session_state (see language_selector() below). The English
string IS the key, by construction, so there is nothing to keep in sync and
no risk of a stale/missing translation silently falling back to a raw key
name -- if a call site is translated, both languages are right there next
to each other in the code.

Language choice lives in st.session_state["awsre_lang"], which Streamlit's
st.navigation()/st.Page() keeps alive across page switches within the same
browser session, so picking a language once (e.g. in the sidebar from
app.py) applies to every page without needing to re-select it.
"""
from __future__ import annotations

import streamlit as st

LANG_STATE_KEY = "awsre_lang"
DEFAULT_LANG = "az"  # dissertation + primary audience is Azerbaijani

LANGUAGE_LABELS = {
    "az": "🇦🇿 Azərbaycan",
    "en": "🇬🇧 English",
}


def get_lang() -> str:
    """Current UI language code ("az" or "en")."""
    return st.session_state.get(LANG_STATE_KEY, DEFAULT_LANG)


def set_lang(lang: str) -> None:
    if lang in LANGUAGE_LABELS:
        st.session_state[LANG_STATE_KEY] = lang


def t(en: str, az: str) -> str:
    """Return `az` if the current UI language is Azerbaijani, else `en`.

    Both languages are always passed at the call site -- there is no
    external lookup table, so a missing translation is impossible: if the
    code compiles, both strings exist.
    """
    return az if get_lang() == "az" else en


def language_selector(container=None, horizontal: bool = True) -> str:
    """Render the AZ/EN toggle and return the (possibly just-changed)
    language code. Call this once per page run, before building any text
    that should react to a language switch on the same rerun (Streamlit
    reruns the whole script top-to-bottom on every widget interaction, so
    as long as this is called before other t(...) calls that feed into
    widget labels created later in the same run, the switch is immediate).
    """
    target = container if container is not None else st.sidebar
    codes = list(LANGUAGE_LABELS.keys())
    current = get_lang()
    choice = target.radio(
        "🌐 Dil / Language",
        codes,
        index=codes.index(current) if current in codes else 0,
        format_func=lambda c: LANGUAGE_LABELS[c],
        key="awsre_lang_widget",
        horizontal=horizontal,
    )
    set_lang(choice)
    return choice
