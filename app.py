"""
app.py — Membre 3
Interface Streamlit du tuteur Study Buddy.
Lance avec : streamlit run app.py
"""

import streamlit as st

# ── Imports des modules des autres membres ───────────────────────────────────
from rag_pipeline import charger_document, decouper_chunks, creer_base_vectorielle, rechercher
from ai_engine import (
    agent_diagnostiqueur,
    analyser_niveau,
    agent_tuteur_socratique,
    agent_evaluateur,
    generer_indice,
)
from session_manager import (
    initialiser_session,
    ajouter_message,
    incrementer_indice,
    reinitialiser_indice,
    get_historique_formate,
    compter_echanges,
)

# ════════════════════════════════════════════════════════════════════════════
# Configuration de la page
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Study Buddy 📚",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# Initialisation de la session (une seule fois)
# ════════════════════════════════════════════════════════════════════════════
if "session" not in st.session_state:
    st.session_state.session = initialiser_session()

session = st.session_state.session


# ════════════════════════════════════════════════════════════════════════════
# CSS personnalisé
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .niveau-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .indice-box {
        background: #3d3000;
        border-left: 4px solid #ffc107;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .eval-box {
        background: #003344;
        border-left: 4px solid #17a2b8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload du document + statut
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📄 Document de cours")

    fichier_uploade = st.file_uploader(
        "Uploade ton PDF ou DOCX",
        type=["pdf", "docx"],
        help="Le document sera découpé et indexé automatiquement."
    )

    if fichier_uploade is not None and not session["document_charge"]:
        with st.spinner("📥 Chargement et indexation du document…"):
            try:
                # Sauvegarde temporaire du fichier uploadé
                import tempfile, os
                suffix = "." + fichier_uploade.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(fichier_uploade.read())
                    chemin_tmp = tmp.name

                # Pipeline RAG : Membre 1
                texte = charger_document(chemin_tmp)
                chunks = decouper_chunks(texte)
                base = creer_base_vectorielle(chunks)

                session["base_vectorielle"] = base
                session["document_charge"] = True
                os.unlink(chemin_tmp)

                st.success(f"✅ {len(chunks)} passages indexés !")
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement : {e}")

    if session["document_charge"]:
        st.success("✅ Document chargé")
        n_vecteurs = session["base_vectorielle"]["index"].ntotal
        st.caption(f"🔢 {n_vecteurs} passages dans la base")
    else:
        st.warning("⚠️ Aucun document chargé")

    st.divider()

    # Statut de la session
    st.header("📊 Statut")
    if session["niveau_etudiant"]:
        couleurs = {"Débutant": "#28a745", "Intermédiaire": "#fd7e14", "Avancé": "#dc3545"}
        couleur = couleurs.get(session["niveau_etudiant"], "#6c757d")
        st.markdown(
            f'Niveau : <span class="niveau-badge" style="background:{couleur}20; color:{couleur};">'
            f'{session["niveau_etudiant"]}</span>',
            unsafe_allow_html=True
        )
    else:
        st.caption("Niveau : Non évalué")

    echanges = compter_echanges(session)
    st.metric("Tours de conversation", echanges)

    if session["niveau_indice"] > 0:
        st.metric("Niveau d'indice actuel", f"{session['niveau_indice']}/3")

    st.divider()

    # Réinitialiser la session
    if st.button("🔄 Nouvelle session", use_container_width=True):
        st.session_state.session = initialiser_session()
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# CONTENU PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">📚 Study Buddy</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ton tuteur IA qui te guide sans te donner les réponses directement</div>',
    unsafe_allow_html=True
)

# ── Affichage de l'historique de conversation ────────────────────────────────
historique_streamlit = get_historique_formate(session)
for msg in historique_streamlit:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Zone de saisie principale ────────────────────────────────────────────────
if not session["document_charge"]:
    st.info("👈 Commence par uploader un document de cours dans la barre latérale.")
else:
    # Phase 1 : Diagnostic
    if session["phase"] == "diagnostic" and not session["sujet"]:
        st.markdown("### 🎯 Quel sujet veux-tu réviser ?")
        sujet_input = st.text_input(
            "Tape ton sujet ou ta question :",
            placeholder="Ex : la photosynthèse, les équations du 2e degré…",
            key="sujet_input"
        )
        if st.button("Commencer la session", type="primary") and sujet_input.strip():
            session["sujet"] = sujet_input.strip()
            with st.spinner("🔍 Diagnostic en cours…"):
                question_diag = agent_diagnostiqueur(session["sujet"])
            ajouter_message(session, "tuteur", question_diag)
            st.rerun()

    # Phase tutoring ou diagnostic (question de niveau posée)
    elif session["sujet"]:
        # Colonne principale pour le chat
        prompt = st.chat_input("Ta réponse…")

        if prompt:
            # Affiche le message de l'étudiant
            with st.chat_message("user"):
                st.markdown(prompt)
            ajouter_message(session, "etudiant", prompt)

            with st.spinner("🧠 Réflexion en cours…"):

                # Si on est encore en phase diagnostic (niveau pas encore déterminé)
                if session["phase"] == "diagnostic" and not session["niveau_etudiant"]:
                    # Analyse le niveau
                    resultat_niveau = analyser_niveau(session["sujet"], prompt)
                    # Extrait le niveau (cherche "NIVEAU : Débutant/Intermédiaire/Avancé")
                    for ligne in resultat_niveau.split("\n"):
                        if "NIVEAU" in ligne:
                            for niveau in ["Débutant", "Intermédiaire", "Avancé"]:
                                if niveau in ligne:
                                    session["niveau_etudiant"] = niveau
                                    break
                    session["phase"] = "tutoring"

                # Recherche les passages pertinents dans le document
                resultats = rechercher(prompt, session["base_vectorielle"], top_k=4)
                chunks_pertinents = [chunk for chunk, _ in resultats]

                # Agent tuteur socratique
                reponse_tuteur = agent_tuteur_socratique(
                    prompt,
                    chunks_pertinents,
                    session["historique"][:-1],  # historique sans le dernier message (déjà ajouté)
                )

            ajouter_message(session, "tuteur", reponse_tuteur)

            with st.chat_message("assistant"):
                st.markdown(reponse_tuteur)

            st.rerun()

        # ── Boutons d'action (indice + évaluation) ───────────────────────────
        if len(session["historique"]) >= 2:
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                niveau_actuel = session["niveau_indice"]
                label_indice = f"💡 Indice ({niveau_actuel}/3)" if niveau_actuel > 0 else "💡 Demander un indice"
                indice_desactive = niveau_actuel >= 3

                if st.button(label_indice, disabled=indice_desactive, use_container_width=True):
                    nouveau_niveau = incrementer_indice(session)
                    dernier_message_etudiant = next(
                        (m["message"] for m in reversed(session["historique"])
                         if m["role"] == "etudiant"), session["sujet"]
                    )
                    with st.spinner(f"Génération de l'indice niveau {nouveau_niveau}…"):
                        indice = generer_indice(dernier_message_etudiant, nouveau_niveau)

                    st.markdown(
                        f'<div class="indice-box">💡 <strong>Indice {nouveau_niveau}/3</strong><br>{indice}</div>',
                        unsafe_allow_html=True
                    )

            with col2:
                if st.button("📝 Évaluer ma réponse", use_container_width=True):
                    dernier_message_etudiant = next(
                        (m["message"] for m in reversed(session["historique"])
                         if m["role"] == "etudiant"), ""
                    )
                    if dernier_message_etudiant:
                        with st.spinner("Évaluation en cours…"):
                            evaluation = agent_evaluateur(dernier_message_etudiant, session["sujet"])

                        st.markdown(
                            f'<div class="eval-box">📊 <strong>Évaluation</strong><br><pre style="white-space:pre-wrap;font-family:inherit;">{evaluation}</pre></div>',
                            unsafe_allow_html=True
                        )
                        # Si la réponse est bonne, réinitialise l'indice
                        if "Avancé" in evaluation or "Intermédiaire" in evaluation:
                            reinitialiser_indice(session)

            with col3:
                st.caption(
                    f"📄 Sujet : **{session['sujet']}** | "
                    f"Phase : **{session['phase']}** | "
                    f"Niveau : **{session['niveau_etudiant'] or 'En cours…'}**"
                )