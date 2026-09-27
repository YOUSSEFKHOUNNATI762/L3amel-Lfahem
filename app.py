# -*- coding: utf-8 -*-
"""
app.py — L3amel Lfahem (العامل الفاهم)
Enterprise Multilingual RAG & Document Intelligence Platform — point d'entrée Streamlit.
"""

import os
import json
import tempfile
import datetime as dt

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

from styles import inject_css, theme_toggle_button, status_dot, badge, LOGO_SVG
from rag_engine import (
    FinOpsGateway, extract_document, build_chunks, SearchIndex,
    mask_pii, detect_prompt_injection, compute_confidence, build_system_prompt,
    log_audit, get_audit_log, log_feedback, get_feedback_log, pdf_diff,
    register_tables_duckdb, text_to_sql, export_docx, export_pdf,
    MODEL_FAST, MODEL_PRO,
)
from agents import (
    cascade_route, call_llm_with_fallback, multi_agent_deliberation,
    analyze_sentiment, translate_citation, regulatory_compliance_check,
)

st.set_page_config(page_title="L3amel Lfahem", page_icon="⚖️", layout="wide")
inject_css()

# ------------------------------------------------------------------------------
# ÉTAT DE SESSION
# ------------------------------------------------------------------------------
defaults = {
    "data_rooms": {}, "current_room": None, "chat_history": [],
    "role": "Lecteur", "target_lang": "Auto", "gateway": None,
    "last_answer": None, "last_sources": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ROOM_ACCESS_RANK = {"public": 0, "RH": 1, "Finance": 1, "Admin": 2}
ROLE_MAX_RANK = {"Lecteur": 0, "RH": 1, "Finance": 1, "Admin": 2}


def visible_rooms():
    role = st.session_state["role"]
    out = []
    for name, room in st.session_state["data_rooms"].items():
        access = room.get("access", "public")
        if role == "Admin" or access == "public" or access == role:
            out.append(name)
    return out


# ==============================================================================
# SIDEBAR — COCKPIT D'ADMINISTRATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">{LOGO_SVG}'
        f'<div><div class="app-title">L3amel Lfahem</div>'
        f'<div class="app-subtitle">العامل الفاهم — RAG Platform</div></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input("🔑 Clé API Groq", type="password", value=os.getenv("GROQ_API_KEY", ""))
    if api_key and st.session_state["gateway"] is None:
        st.session_state["gateway"] = FinOpsGateway(api_key=api_key)
    elif not api_key:
        st.session_state["gateway"] = None

    st.markdown("**Status Hub**")
    status_dot("LLM Gateway Ready", ok=st.session_state["gateway"] is not None)
    status_dot("VectorDB Connected", ok=len(st.session_state["data_rooms"]) > 0)
    status_dot("Local Engine (fallback)", ok=True)
    st.divider()

    st.markdown("**Upload Center — Data Room**")
    room_name = st.text_input("Nom du data room", value="Code du travail")
    access_level = st.selectbox("Confidentialité", ["public", "RH", "Finance", "Admin"])
    uploaded = st.file_uploader("Déposer un ou plusieurs PDF", type=["pdf"], accept_multiple_files=True)

    if st.button("⚙️ Traiter les documents", use_container_width=True, disabled=not uploaded):
        progress = st.progress(0, text="Extraction en cours...")
        all_pages, all_tables, all_meta, paths = [], [], [], []
        for i, f in enumerate(uploaded):
            tmp_path = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp_path, "wb") as out:
                out.write(f.getbuffer())
            doc = extract_document(tmp_path)
            all_pages.extend(doc["pages"])
            all_tables.extend(doc["tables"])
            all_meta.append(doc["metadata"])
            paths.append(tmp_path)
            progress.progress(int((i + 1) / len(uploaded) * 60), text=f"Extraction : {f.name}")

        chunks = build_chunks(all_pages, data_room=room_name)
        progress.progress(80, text="Indexation vectorielle + BM25...")
        index = SearchIndex(chunks)
        progress.progress(100, text="Terminé")

        st.session_state["data_rooms"][room_name] = {
            "pages": all_pages, "tables": all_tables, "metadata": all_meta,
            "chunks": chunks, "index": index, "access": access_level, "paths": paths,
            "anomalies": [a for p in [extract_document(p)["anomalies"] for p in paths] for a in p],
        }
        st.session_state["current_room"] = room_name
        st.success(f"Data room « {room_name} » prêt ({len(chunks)} chunks, {len(all_tables)} tables).")

    rooms = visible_rooms()
    if rooms:
        st.session_state["current_room"] = st.selectbox("Data room actif", rooms,
                                                          index=rooms.index(st.session_state["current_room"])
                                                          if st.session_state["current_room"] in rooms else 0)
    st.divider()

    st.markdown("**Switcher linguistique**")
    st.session_state["target_lang"] = st.selectbox(
        "Langue de réponse forcée", ["Auto", "Français", "Darija", "Arabe classique", "Anglais"]
    )
    multi_doc = st.checkbox("Recherche multi-documents (tous les data rooms visibles)", value=False)
    use_multi_agent = st.checkbox("Mode Multi-Agents (Juridique + Financier + Synthèse)", value=False)

    st.divider()
    st.markdown("**Rôle (RBAC simulation)**")
    st.session_state["role"] = st.selectbox("Rôle utilisateur", ["Lecteur", "RH", "Finance", "Admin"])

    st.divider()
    theme_toggle_button()

    st.divider()
    st.markdown("**Module FinOps**")
    if st.session_state["gateway"]:
        snap = st.session_state["gateway"].snapshot()
        st.markdown(f'<div class="finops-metric"><span>Tokens in/out</span>'
                     f'<span class="finops-value">{snap["input_tokens"]}/{snap["output_tokens"]}</span></div>'
                     f'<div class="finops-metric"><span>Coût session</span>'
                     f'<span class="finops-value">${snap["cost_usd"]}</span></div>'
                     f'<div class="finops-metric"><span>Latence dernier appel</span>'
                     f'<span class="finops-value">{snap["last_latency_ms"]} ms</span></div>',
                     unsafe_allow_html=True)
    else:
        st.caption("Renseignez une clé API Groq pour activer le suivi FinOps.")


# ==============================================================================
# HELPERS
# ==============================================================================

def gather_context(question: str, k: int = 5):
    gateway = st.session_state["gateway"]
    rooms = visible_rooms() if multi_doc else [st.session_state["current_room"]]
    candidats = []
    for r in rooms:
        if not r:
            continue
        idx: SearchIndex = st.session_state["data_rooms"][r]["index"]
        candidats.extend(idx.hybrid_search(question, gateway, k=k))
    candidats.sort(key=lambda c: -c.get("_rerank_score", 0))
    top = candidats[:k]
    contexte = "\n".join(
        f"[Source {i+1} — {c['source_file']} | Page {c['page']} | {c['article']}]\n{c['texte_parent']}"
        for i, c in enumerate(top)
    )
    return top, contexte


def render_citations(sources):
    for i, s in enumerate(sources, 1):
        st.markdown(f'<span class="citation-badge">📄 {i}. Page {s["page"]} · {s["article"]}</span>',
                    unsafe_allow_html=True)


# ==============================================================================
# WORKSPACES
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chatbot & Lecteur", "📊 Analytics & Diagrammes",
    "🔍 Analyse différentielle & Conformité", "🗂️ Audit & Exports",
])

# --- WORKSPACE 1 : CHATBOT & READER MODE ------------------------------------
with tab1:
    col_chat, col_reader = st.columns([1, 1])

    with col_chat:
        st.markdown("#### Assistant L3amel Lfahem")

        audio = st.audio_input("🎙️ Poser une question à l'oral")
        question_vocale = None
        if audio and st.session_state["gateway"]:
            tmp_audio = os.path.join(tempfile.gettempdir(), "voice_query.wav")
            with open(tmp_audio, "wb") as f:
                f.write(audio.getbuffer())
            question_vocale = st.session_state["gateway"].transcribe_audio(tmp_audio)
            st.caption(f"🗣️ Transcrit : {question_vocale}")

        question = st.chat_input("Écrivez votre question (FR / Darija / العربية / English)...")
        question = question_vocale or question

        for msg in st.session_state["chat_history"]:
            cls = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
            st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg["role"] == "assistant" and msg.get("sources"):
                render_citations(msg["sources"])
                conf = msg.get("confidence", {})
                st.markdown(
                    f'<div class="confidence-bar-bg"><div class="confidence-bar-fill" '
                    f'style="width:{conf.get("score",0)}%;background:'
                    f'{"#15803D" if conf.get("kind")=="success" else "#B45309" if conf.get("kind")=="warning" else "#B91C1C"};">'
                    f'</div></div><span style="font-size:11px;">{conf.get("label","")} ({conf.get("score",0)}%)</span>',
                    unsafe_allow_html=True,
                )

        if question:
            if not st.session_state["current_room"]:
                st.warning("Téléversez d'abord un document dans un data room (menu latéral).")
            elif not st.session_state["gateway"]:
                st.warning("Renseignez votre clé API Groq dans le menu latéral.")
            else:
                is_injection, reason = detect_prompt_injection(question)
                masked_question, redactions = mask_pii(question)
                st.session_state["chat_history"].append({"role": "user", "content": question})

                if is_injection:
                    reponse_text = ("⚠️ Cette requête contient un motif potentiellement "
                                     "malveillant et a été bloquée par le module anti-injection.")
                    sources, confiance = [], {"score": 0, "label": "Bloqué", "kind": "danger"}
                else:
                    sources, contexte = gather_context(masked_question)
                    langue_cible = st.session_state["target_lang"]
                    lang_hint = "" if langue_cible == "Auto" else f"\nRéponds obligatoirement en {langue_cible}."

                    if use_multi_agent:
                        result = multi_agent_deliberation(masked_question, contexte, langue_cible,
                                                            st.session_state["gateway"])
                        reponse_text = result["reponse_finale"]
                        model_used = "multi-agent"
                    else:
                        model = cascade_route(masked_question)
                        sys_prompt = build_system_prompt(lang_hint)
                        user_prompt = f"CONTEXTE :\n{contexte}\n\nQUESTION : {masked_question}"
                        result = call_llm_with_fallback(
                            st.session_state["gateway"],
                            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                            model=model,
                        )
                        reponse_text = result["text"]
                        model_used = result.get("model", model)

                    confiance = compute_confidence(sources)
                    log_audit(
                        user=st.session_state["role"], question=question, model=model_used,
                        tokens_in=result.get("input_tokens", 0) if not use_multi_agent else 0,
                        tokens_out=result.get("output_tokens", 0) if not use_multi_agent else 0,
                        cost_usd=result.get("cost_usd", 0) if not use_multi_agent else result.get("cout_total", 0),
                        latency_ms=result.get("latency_ms", 0) if not use_multi_agent else 0,
                        doc_source=st.session_state["current_room"], confidence=confiance["score"],
                        redactions=len(redactions),
                    )

                st.session_state["chat_history"].append({
                    "role": "assistant", "content": reponse_text,
                    "sources": sources, "confidence": confiance,
                })
                st.session_state["last_answer"] = reponse_text
                st.session_state["last_sources"] = sources
                st.rerun()

        if st.session_state["chat_history"] and st.session_state["chat_history"][-1]["role"] == "assistant":
            c1, c2, c3 = st.columns(3)
            if c1.button("👍 Utile"):
                log_feedback(st.session_state["chat_history"][-2]["content"],
                             st.session_state["chat_history"][-1]["content"], 1)
                st.toast("Merci pour votre retour !")
            if c2.button("👎 Pas utile"):
                log_feedback(st.session_state["chat_history"][-2]["content"],
                             st.session_state["chat_history"][-1]["content"], -1)
                st.toast("Merci, nous en tiendrons compte.")
            if c3.button("🔊 Écouter la réponse"):
                try:
                    from gtts import gTTS
                    tts_path = os.path.join(tempfile.gettempdir(), "reponse.mp3")
                    gTTS(text=st.session_state["last_answer"][:800], lang="fr").save(tts_path)
                    st.audio(tts_path)
                except Exception as e:
                    st.error(f"Synthèse vocale indisponible : {e}")

    with col_reader:
        st.markdown("#### Lecteur PDF — Citation Deep-Link")
        if st.session_state["current_room"]:
            room = st.session_state["data_rooms"][st.session_state["current_room"]]
            if st.session_state["last_sources"]:
                default_page = st.session_state["last_sources"][0]["page"]
            else:
                default_page = 1
            page_choisie = st.number_input("Page à afficher", min_value=1,
                                            max_value=max(1, len(room["pages"])), value=int(default_page))
            try:
                from pdf2image import convert_from_path
                img = convert_from_path(room["paths"][0], first_page=page_choisie, last_page=page_choisie)
                if img:
                    st.image(img[0], use_container_width=True)
            except Exception:
                page_data = next((p for p in room["pages"] if p["page"] == page_choisie), None)
                st.info("Aperçu image indisponible (poppler non installé) — texte extrait ci-dessous :")
                st.write(page_data["texte"] if page_data else "")
        else:
            st.caption("Aucun document chargé.")

# --- WORKSPACE 2 : ANALYTICS & DIAGRAMS -------------------------------------
with tab2:
    if not st.session_state["current_room"]:
        st.info("Chargez un document pour activer ce workspace.")
    else:
        room = st.session_state["data_rooms"][st.session_state["current_room"]]
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Auto-Dashboarding (tables extraites)")
            if room["tables"]:
                labels = [t["table_id"] for t in room["tables"]]
                choix = st.selectbox("Table détectée", labels)
                df = next(t["dataframe"] for t in room["tables"] if t["table_id"] == choix)
                st.dataframe(df, use_container_width=True)
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    fig = px.bar(df, y=num_cols[0], template="simple_white")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Text-to-SQL (DuckDB)")
                sql_question = st.text_input("Question analytique sur les tables (ex: total, moyenne...)")
                if st.button("Exécuter en SQL") and sql_question and st.session_state["gateway"]:
                    con, noms = register_tables_duckdb(room["tables"])
                    sql, df_res, err = text_to_sql(sql_question, con, noms, st.session_state["gateway"])
                    st.code(sql, language="sql")
                    if err:
                        st.error(err)
                    else:
                        st.dataframe(df_res, use_container_width=True)
            else:
                st.caption("Aucun tableau numérique détecté dans ce data room.")

        with colB:
            st.markdown("#### Carte mentale (Mermaid.js)")
            if st.button("🧠 Générer une carte du document") and st.session_state["gateway"]:
                echantillon = " ".join(p["texte"] for p in room["pages"][:5])[:2500]
                prompt = (
                    "Génère un diagramme Mermaid (syntaxe 'graph TD') représentant la structure "
                    "hiérarchique des thèmes principaux de ce texte juridique. Réponds uniquement "
                    f"avec le code Mermaid brut, sans balises markdown.\n\n{echantillon}"
                )
                res = st.session_state["gateway"].chat([{"role": "user", "content": prompt}], model=MODEL_FAST)
                mermaid_code = res["text"].replace("```mermaid", "").replace("```", "").strip()
                components.html(f"""
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                    <div class="mermaid">{mermaid_code}</div>
                    <script>mermaid.initialize({{startOnLoad:true}});</script>
                """, height=420, scrolling=True)

            st.markdown("#### Quiz d'évaluation")
            if st.button("📝 Générer un QCM (5 questions)") and st.session_state["gateway"]:
                echantillon = " ".join(p["texte"] for p in room["pages"][:6])[:3000]
                prompt = (
                    'Génère un QCM de 5 questions au format JSON strict : '
                    '[{"question":"...","options":["A","B","C","D"],"reponse_correcte":0}] '
                    f"à partir de ce texte :\n\n{echantillon}"
                )
                res = st.session_state["gateway"].chat([{"role": "user", "content": prompt}], model=MODEL_PRO)
                try:
                    quiz = json.loads(res["text"])
                    st.session_state["quiz"] = quiz
                except Exception:
                    st.error("Le modèle n'a pas renvoyé un JSON valide — réessayez.")

            if "quiz" in st.session_state:
                score = 0
                for i, q in enumerate(st.session_state["quiz"]):
                    choix = st.radio(q["question"], q["options"], key=f"quiz_{i}")
                    if q["options"].index(choix) == q["reponse_correcte"]:
                        score += 1
                st.markdown(f"**Score : {score} / {len(st.session_state['quiz'])}**")

# --- WORKSPACE 3 : DIFFERENTIAL ANALYSIS & COMPLIANCE -----------------------
with tab3:
    st.markdown("#### Analyse différentielle de PDF")
    c1, c2 = st.columns(2)
    file_a = c1.file_uploader("Version A", type=["pdf"], key="diff_a")
    file_b = c2.file_uploader("Version B", type=["pdf"], key="diff_b")

    if file_a and file_b and st.button("Comparer les versions"):
        path_a = os.path.join(tempfile.gettempdir(), "v_a_" + file_a.name)
        path_b = os.path.join(tempfile.gettempdir(), "v_b_" + file_b.name)
        with open(path_a, "wb") as f:
            f.write(file_a.getbuffer())
        with open(path_b, "wb") as f:
            f.write(file_b.getbuffer())
        doc_a, doc_b = extract_document(path_a), extract_document(path_b)
        diff_result = pdf_diff(doc_a["pages"], doc_b["pages"])
        for d in diff_result:
            if d["modifie"]:
                st.markdown(f"**Page {d['page']}** — modifications détectées", help="Vert = ajout, Rouge = suppression")
                st.markdown(f'<div class="custom-card">{d["html"][:3000]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Vérification de conformité réglementaire")
    contrat_file = st.file_uploader("Document à vérifier (contrat, règlement interne...)", type=["pdf"], key="compliance")
    if contrat_file and st.button("Lancer l'analyse de conformité") and st.session_state["current_room"]:
        path_c = os.path.join(tempfile.gettempdir(), "compliance_" + contrat_file.name)
        with open(path_c, "wb") as f:
            f.write(contrat_file.getbuffer())
        doc_c = extract_document(path_c)
        texte_complet = " ".join(p["texte"] for p in doc_c["pages"])
        room = st.session_state["data_rooms"][st.session_state["current_room"]]
        ref_chunks, _ = gather_context("obligations et clauses générales du contrat de travail", k=6)
        rapport = regulatory_compliance_check(texte_complet, ref_chunks, st.session_state["gateway"])
        st.markdown(f'<div class="custom-card">{rapport}</div>', unsafe_allow_html=True)

# --- WORKSPACE 4 : AUDIT LOG & EXECUTIVE EXPORTS ----------------------------
with tab4:
    st.markdown("#### Registre d'audit (conformité SOC2-ready)")
    st.dataframe(get_audit_log(), use_container_width=True)

    st.markdown("#### Boucle RLHF — Feedback utilisateurs")
    fb = get_feedback_log()
    if not fb.empty:
        st.metric("Note moyenne", round(fb["note"].mean(), 2))
        st.dataframe(fb, use_container_width=True)
    else:
        st.caption("Aucun feedback enregistré pour le moment.")

    st.markdown("#### Export exécutif")
    if st.session_state["last_answer"]:
        col1, col2 = st.columns(2)
        docx_bytes = export_docx("Rapport L3amel Lfahem",
                                  st.session_state["chat_history"][-2]["content"] if len(st.session_state["chat_history"]) >= 2 else "",
                                  st.session_state["last_answer"], st.session_state["last_sources"])
        pdf_bytes = export_pdf("Rapport L3amel Lfahem",
                                st.session_state["chat_history"][-2]["content"] if len(st.session_state["chat_history"]) >= 2 else "",
                                st.session_state["last_answer"], st.session_state["last_sources"])
        col1.download_button("⬇️ Exporter en Word", docx_bytes, file_name="rapport_l3amel_lfahem.docx")
        col2.download_button("⬇️ Exporter en PDF", pdf_bytes, file_name="rapport_l3amel_lfahem.pdf")
    else:
        st.caption("Posez une question dans le Workspace 1 pour activer l'export.")
