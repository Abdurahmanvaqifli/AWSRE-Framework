"""
AWSRE Framework
Main Streamlit Application (page router)

Single page: the simple Protect & Verify workflow (pages/0_Home.py).

Uses Streamlit's explicit st.navigation()/st.Page() API instead of the old
implicit "every file in pages/ becomes a sidebar entry" behavior. Once
st.navigation() is called here, ONLY the pages listed below appear in the
sidebar or are reachable at all -- the framework does not fall back to
auto-discovering the rest of pages/.

The older pages/1_Dashboard.py, pages/2_Recommendation.py,
pages/4_Benchmark.py and pages/5_Strategy_Explorer.py files still exist on
disk (kept for reference / possible future reuse) but are intentionally not
registered here anymore, per the decision to trim the app down to the pages
that matter: the ML-based Intelligent Mode recommender, manual Embedding,
and the new literature-based Comparison page.

Note: st.set_page_config() may only be called ONCE per app run. It is
called here, and NOT inside the individual page files (each page file
guards its own st.set_page_config() call in a try/except so it still works
if run standalone during development).

Bilingual UI (AZ/EN): the language toggle lives here, in the sidebar,
before the st.Page(...) objects are built, so a language switch is
reflected in the sidebar page titles on the very same rerun. See i18n.py
for how it works; every page imports `t()` from there to translate its own
text using st.session_state["awsre_lang"], which Streamlit keeps alive
across page switches within one browser session.
"""

import streamlit as st

from i18n import language_selector, t

st.set_page_config(
    page_title="AWSRE Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

language_selector()

if not hasattr(st, "navigation") or not hasattr(st, "Page"):
    st.error(
        t(
            "This version of Streamlit does not support st.navigation() / "
            "st.Page() (added in Streamlit 1.36+). Please upgrade with:\n\n"
            "pip install --upgrade streamlit\n\n"
            "and restart the app. Until then, the sidebar will show every "
            "page file in pages/ instead of only the 3 consolidated pages.",
            "Streamlit-in bu versiyası st.navigation() / st.Page() "
            "funksiyalarını dəstəkləmir (Streamlit 1.36+ tələb olunur). "
            "Zəhmət olmasa yeniləyin:\n\n"
            "pip install --upgrade streamlit\n\n"
            "və tətbiqi yenidən başladın. Bu edilənə qədər, yan paneldə "
            "yalnız 3 birləşdirilmiş səhifə əvəzinə pages/ qovluğundakı "
            "bütün fayllar göstəriləcək.",
        )
    )
    st.stop()

home_page = st.Page(
    "pages/0_Home.py",
    title=t("Protect & Verify", "Qoru və Yoxla"),
    icon="🛡️",
    default=True,
)

# One page only. The recommendation, the 15-strategy view and the
# comparison with a fixed strategy now appear inside Protect's result, so
# pages/6_Intelligent_Mode.py, pages/3_Embedding.py and pages/7_Comparison.py
# are no longer registered (the files stay on disk for reference).
navigation = st.navigation([home_page], position="hidden")

navigation.run()
