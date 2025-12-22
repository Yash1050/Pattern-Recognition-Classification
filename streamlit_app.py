#!/usr/bin/env python3
"""
Minimal-but-polished Streamlit UI for the existing RAGEngine.

Notes:
- Reuses RAGEngine exactly as-is (no RAG logic reimplementation).
- Displays only the final answer by default.
- Optional debug section shows internal pipeline details.
"""

import streamlit as st

from rag_engine import RAGEngine


@st.cache_resource(show_spinner=False)
def get_engine():
    # Uses the existing defaults (intent model, Qdrant creds, OpenAI key from env/.env)
    return RAGEngine()


def main():
    st.set_page_config(page_title="RAG Engine Demo", page_icon="🔎", layout="wide")

    # --- Light styling for a more engaging UI
    st.markdown(
        """
        <style>
        /* Page background and font tweaks */
        .stApp {
            background: radial-gradient(circle at 20% 20%, #202b44 0, #111827 35%, #0b1220 100%);
            color: #e5e7eb;
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
        }
        /* Card-like container */
        .glass {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 20px 24px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            font-size: 0.85rem;
        }
        .pill.ok    { color: #10b981; }
        .pill.warn  { color: #f59e0b; }
        .pill.fail  { color: #ef4444; }
        textarea, input, .stTextInput, .stTextArea {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Hero header
    st.markdown(
        """
        <div class="glass" style="margin-bottom: 1rem;">
            <h1 style="margin:0 0 0.4rem 0;">🔎 RAG Engine</h1>
            <p style="margin:0; color:#cbd5f5;">
                Ask a question and get a grounded answer. Toggle debug to inspect the pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_meta = st.columns([3, 1.2])

    with col_input:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        query = st.text_area("Your question", value="", height=140, label_visibility="collapsed")
        debug = st.checkbox("Show debug details")
        run_btn = st.button("Run", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_meta:
        st.markdown(
            """
            <div class="glass">
                <h4 style="margin-top:0;">💡 Tips</h4>
                <ul style="padding-left: 1rem; color:#cbd5f5;">
                    <li>Provide clear questions</li>
                    <li>Use debug to inspect intent & retrieval</li>
                    <li>Fallback engages when evidence is weak</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_btn:
        query_clean = query.strip()
        if not query_clean:
            st.warning("Please enter a question.")
            return

        with st.spinner("Running RAG pipeline..."):
            try:
                rag = get_engine()
                response = rag.answer_query(query_clean, verbose=False)
            except Exception as e:
                st.error(f"Error running RAG engine: {e}")
                return

        # Final answer (always shown)
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Answer")
        st.write(response.get("generation", {}).get("answer", "No answer produced."))
        st.markdown("</div>", unsafe_allow_html=True)

        # Quick outcome pills
        intent = response.get("intent", {})
        retrieval = response.get("retrieval", {})
        fallback_used = response.get("fallback_used", False)
        chunks_retrieved = retrieval.get("chunks_retrieved", 0)
        intent_name = intent.get("name", "?")

        pill_color = "warn" if fallback_used else "ok"
        fallback_text = "Fallback used" if fallback_used else "Grounded"

        st.markdown(
            f"""
            <div style="margin-top:0.5rem; display:flex; gap:12px; flex-wrap:wrap;">
              <span class="pill {pill_color}">⚙️ {fallback_text}</span>
              <span class="pill">🧭 Intent: {intent_name}</span>
              <span class="pill">📚 Chunks: {chunks_retrieved}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Optional debug info
        if debug:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("Debug Info")
            intent = response.get("intent", {})
            retrieval = response.get("retrieval", {})

            st.markdown(f"- **Intent**: {intent.get('name', '?')} (ID: {intent.get('id', '?')})")
            st.markdown(f"- **Fallback Used**: {response.get('fallback_used', False)}")
            st.markdown(f"- **Chunks Retrieved**: {retrieval.get('chunks_retrieved', 0)}")
            st.markdown(f"- **Context Length**: {retrieval.get('context_length', 0)} chars")

            top_chunks = retrieval.get("top_chunks", []) or []
            if top_chunks:
                st.markdown("**Top Retrieved Sources:**")
                for i, chunk in enumerate(top_chunks, 1):
                    source = chunk.get("source", "unknown")
                    score = chunk.get("score", 0.0)
                    snippet = chunk.get("text", "")
                    st.markdown(f"{i}. **Source**: {source} | **Score**: {score:.4f}")
                    st.caption(snippet)
            else:
                st.markdown("_No retrieved chunks available._")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

