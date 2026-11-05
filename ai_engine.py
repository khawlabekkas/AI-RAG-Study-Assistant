import os
from typing import List, Dict

from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Global model
MODEL = "llama-3.3-70b-versatile"

# Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _appeler_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """
    Appelle le modèle Groq et retourne la réponse texte.
    """
    reponse = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return reponse.choices[0].message.content


def agent_diagnostiqueur(question: str) -> str:
    """
    Pose une question courte pour évaluer le niveau de l'étudiant.
    """
    system_prompt = """Tu es un agent diagnostiqueur pédagogique.
Ton rôle est d'évaluer le niveau de l'étudiant AVANT de commencer à lui enseigner.

Règles STRICTES :
- Pose exactement 1 question COURTE sur le sujet demandé.
- La question doit permettre de distinguer un débutant d'un étudiant avancé.
- Ne donne AUCUNE explication ni réponse. Pose seulement la question.
- Commence directement par la question, sans introduction.
- Langue : français."""

    user_prompt = f"L'étudiant veut apprendre sur : {question}"

    return _appeler_llm(system_prompt, user_prompt, temperature=0.5)


def analyser_niveau(matiere: str, reponses_etudiant: str) -> str:
    """
    Analyse la réponse de l'étudiant et retourne son niveau.
    """
    system_prompt = f"""Tu es un agent diagnostiqueur pédagogique strict.
La matière évaluée est : {matiere}

RÈGLE PRINCIPALE : En cas de doute, classe TOUJOURS au niveau inférieur.

DÉBUTANT — si l'étudiant :
- Dit "je ne sais pas" ou "je ne connais pas les détails"
- Donne seulement une réponse très vague sans explication
- Montre qu'il ne connaît pas les concepts de base de {matiere}

INTERMÉDIAIRE — si l'étudiant :
- Connaît les concepts de base de {matiere} et les explique correctement
- Utilise des mots simples mais justes
- Manque de précision sur les détails avancés de {matiere}

AVANCÉ — si l'étudiant :
- Utilise le vocabulaire technique précis de {matiere}
- Explique les mécanismes détaillés avec précision
- Montre une vraie maîtrise de {matiere}

Réponds TOUJOURS dans ce format exact :
NIVEAU : [Débutant / Intermédiaire / Avancé]
RAISON : [Une phrase courte]"""

    user_prompt = f"""Matière : {matiere}
Réponses de l'étudiant : {reponses_etudiant}"""

    return _appeler_llm(system_prompt, user_prompt, temperature=0.1)


def agent_tuteur_socratique(
    question: str,
    chunks: List[str],
    historique: List[Dict[str, str]],
) -> str:
    """
    Guide l'étudiant avec la méthode socratique sans donner la réponse directement.
    """
    contexte_docs = "\n\n".join(
        [f"[Extrait {i + 1}] {chunk}" for i, chunk in enumerate(chunks)]
    ) if chunks else "Aucun document disponible."

    historique_texte = "\n".join(
        [f"{msg['role'].upper()} : {msg['message']}" for msg in historique]#transforme l historique en text pour envoyer a llm
    ) if historique else "Début de la conversation."

    system_prompt = """Tu es un tuteur pédagogique qui utilise la méthode socratique.

RÈGLE ABSOLUE : Tu ne donnes JAMAIS la réponse directement.

À la place, tu :
1. Poses des questions qui amènent l'étudiant à réfléchir.
2. Fais des analogies simples avec des exemples du quotidien.
3. Mets en valeur ce que l'étudiant a déjà bien compris.
4. Orientes doucement vers la bonne piste sans la révéler.

Tu t'appuies sur les extraits du document pour guider sans les copier mot pour mot.
Si l'étudiant donne une mauvaise réponse, tu ne dis pas "c'est faux".
Tu demandes : "Qu'est-ce qui te fait penser ça ?"

Langue : français.
Ton : bienveillant et encourageant."""

    user_prompt = f"""QUESTION DE L'ÉTUDIANT : {question}

EXTRAITS DU DOCUMENT :
{contexte_docs}

HISTORIQUE DE LA CONVERSATION :
{historique_texte}

Génère une réponse socratique courte, entre 3 et 5 phrases maximum."""

    return _appeler_llm(system_prompt, user_prompt, temperature=0.7)


def agent_evaluateur(reponse_etudiant: str, question_initiale: str) -> str:
    """
    Évalue la réponse de l'étudiant.
    """
    system_prompt = """Tu es un évaluateur pédagogique strict.

RÈGLE PRINCIPALE :
Évalue uniquement ce que l'étudiant a écrit.

DÉBUTANT :
- Réponse vague
- Peu ou pas de vocabulaire scientifique
- Compréhension très limitée

INTERMÉDIAIRE :
- Concepts de base compris
- Réponse correcte mais incomplète
- Manque de précision

AVANCÉ :
- Vocabulaire précis
- Explication structurée
- Bonne maîtrise du sujet

Réponds TOUJOURS dans ce format :
NIVEAU : [Débutant / Intermédiaire / Avancé]
CE QUI EST BIEN COMPRIS :
- [point 1]
POINTS ENCORE FLOUS :
- [point 1]
SUGGESTION :
[Une phrase courte]"""

    user_prompt = f"""Question : {question_initiale}
Réponse de l'étudiant : {reponse_etudiant}"""

    return _appeler_llm(system_prompt, user_prompt, temperature=0.1)


def generer_indice(question: str, niveau_indice: int) -> str:
    """
    Génère un indice progressif de niveau 1 à 3.
    """
    if niveau_indice not in [1, 2, 3]:
        raise ValueError(
            f"niveau_indice doit être 1, 2 ou 3. Reçu : {niveau_indice}"
        )

    descriptions_niveau = {
        1: "très discret — pose une question de réflexion générale sans mentionner de concept précis",
        2: "modéré — mentionne un concept clé lié à la réponse, mais ne l'explique pas",
        3: "fort — donne une explication partielle très proche de la réponse complète",
    }

    system_prompt = f"""Tu es un tuteur pédagogique bienveillant.
Tu génères un indice de niveau {niveau_indice}/3 pour aider un étudiant bloqué.

Niveau {niveau_indice} signifie : indice {descriptions_niveau[niveau_indice]}.

Règles :
- L'indice doit tenir en 2-3 phrases maximum.
- Commence par une phrase d'encouragement.
- Ne dis jamais explicitement la réponse finale.
- Langue : français."""

    user_prompt = f"""L'étudiant est bloqué sur la question : {question}

Génère un indice de niveau {niveau_indice}/3."""

    return _appeler_llm(system_prompt, user_prompt, temperature=0.6)