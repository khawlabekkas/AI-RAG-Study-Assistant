"""
session_manager.py — Membre 3
Gestion de la mémoire de session : historique, niveau d'indice, état de l'étudiant.
Pas de dépendances externes — utilise uniquement le state Streamlit (st.session_state).
"""

from typing import Dict, Any


# ════════════════════════════════════════════════════════════════════════════
# 1. initialiser_session
# ════════════════════════════════════════════════════════════════════════════
def initialiser_session() -> Dict[str, Any]:
    """
    Crée et retourne une session vide.

    Retourne
    --------
    dict avec les clés :
        - "historique"    : List[Dict]  — messages de la conversation
        - "niveau_indice" : int         — niveau d'indice actuel (0 = pas encore demandé)
        - "niveau_etudiant" : str       — "Débutant" | "Intermédiaire" | "Avancé" | None
        - "base_vectorielle" : dict|None — index FAISS chargé depuis le document
        - "document_charge" : bool      — True si un document a été uploadé et indexé
        - "phase"         : str         — "diagnostic" | "tutoring" (phase courante)
        - "sujet"         : str         — sujet de la session en cours
    """
    return {
        "historique": [],
        "niveau_indice": 0,
        "niveau_etudiant": None,
        "base_vectorielle": None,
        "document_charge": False,
        "phase": "diagnostic",
        "sujet": "",
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. ajouter_message
# ════════════════════════════════════════════════════════════════════════════
def ajouter_message(session: Dict, role: str, message: str) -> None:
    """
    Ajoute un message à l'historique de la session.

    Paramètres
    ----------
    session : dict — la session courante (modifiée en place)
    role    : str  — "etudiant" ou "tuteur"
    message : str  — contenu du message

    Lève
    ----
    ValueError si role n'est pas "etudiant" ou "tuteur".
    """
    if role not in ("etudiant", "tuteur"):
        raise ValueError(f"role doit être 'etudiant' ou 'tuteur'. Reçu : '{role}'")
    session["historique"].append({"role": role, "message": message})


# ════════════════════════════════════════════════════════════════════════════
# 3. incrementer_indice
# ════════════════════════════════════════════════════════════════════════════
def incrementer_indice(session: Dict) -> int:
    """
    Passe au niveau d'indice suivant (max 3).

    Retourne
    --------
    int — le nouveau niveau d'indice (1, 2 ou 3)
    """
    if session["niveau_indice"] < 3:
        session["niveau_indice"] += 1
    return session["niveau_indice"]


# ════════════════════════════════════════════════════════════════════════════
# 4. reinitialiser_indice
# ════════════════════════════════════════════════════════════════════════════
def reinitialiser_indice(session: Dict) -> None:
    """
    Remet le niveau d'indice à 0 (quand la question est résolue).

    Paramètres
    ----------
    session : dict — la session courante (modifiée en place)
    """
    session["niveau_indice"] = 0


# ════════════════════════════════════════════════════════════════════════════
# Utilitaires supplémentaires
# ════════════════════════════════════════════════════════════════════════════
def get_historique_formate(session: Dict) -> list:
    """
    Retourne l'historique dans le format attendu par Streamlit (role/content).
    Utile pour afficher les messages avec st.chat_message().

    Retourne
    --------
    List[Dict] — [{"role": "user"|"assistant", "content": "..."}]
    """
    formate = []
    for msg in session["historique"]:
        role_streamlit = "user" if msg["role"] == "etudiant" else "assistant"
        formate.append({"role": role_streamlit, "content": msg["message"]})
    return formate


def compter_echanges(session: Dict) -> int:
    """Retourne le nombre de tours de conversation (1 tour = 1 message étudiant + 1 tuteur)."""
    return len([m for m in session["historique"] if m["role"] == "etudiant"])
