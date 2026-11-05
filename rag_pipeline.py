import os
import re
from typing import List, Tuple

import fitz                          # PyMuPDF → PDF
from docx import Document            # python-docx → DOCX
from langchain_text_splitters import RecursiveCharacterTextSplitter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("✅ Tous les imports sont OK !")

# Modèle multilingue — fonctionne très bien en français
_EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_embed_model = None

def _get_embed_model() -> SentenceTransformer:
    """Retourne le modèle d'embedding (chargé une seule fois)."""
    global _embed_model
    if _embed_model is None:
        print(f"⏳ Chargement du modèle : {_EMBED_MODEL_NAME}…")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        print("✅ Modèle chargé !")
    return _embed_model

# Pré-chargement
_get_embed_model()

def _lire_pdf(chemin: str) -> str:
    """Extrait le texte de toutes les pages d'un PDF avec PyMuPDF."""
    pages = []
    with fitz.open(chemin) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def _lire_docx(chemin: str) -> str:
    """Extrait le texte de tous les paragraphes d'un DOCX."""
    doc = Document(chemin)
    return "\n".join(para.text for para in doc.paragraphs)


def _nettoyer_texte(texte: str) -> str:
    """Supprime les espaces et sauts de ligne superflus."""
    texte = re.sub(r"[ \t]+", " ", texte)      # espaces multiples → 1
    texte = re.sub(r"\n{3,}", "\n\n", texte)    # > 2 sauts → 2
    return texte.strip()


def charger_document(fichier: str) -> str:
    """
    Lit un fichier PDF ou DOCX et retourne son texte brut nettoyé.

    Paramètres
    ----------
    fichier : str — chemin vers le fichier (.pdf ou .docx)

    Retourne
    --------
    str — texte extrait et nettoyé
    """
    if not os.path.isfile(fichier):
        raise FileNotFoundError(f"Fichier introuvable : {fichier}")

    extension = os.path.splitext(fichier)[1].lower()

    if extension == ".pdf":
        texte = _lire_pdf(fichier)
    elif extension == ".docx":
        texte = _lire_docx(fichier)
    else:
        raise ValueError(f"Extension « {extension} » non supportée. Utilisez .pdf ou .docx.")

    return _nettoyer_texte(texte)


print("✅ charger_document() définie.")

CHEMIN_FICHIER = "plant.pdf"   # ← modifie ici

texte_brut = charger_document(CHEMIN_FICHIER)
print(f"📄 Caractères extraits : {len(texte_brut)}")
print("\n--- Aperçu (500 premiers caractères) ---")
print(texte_brut[:500])

def decouper_chunks(
    texte: str,
    taille_chunk: int = 500,
    chevauchement: int = 100,
) -> List[str]:
    """
    Découpe le texte en morceaux (chunks) avec chevauchement.

    Paramètres
    ----------
    texte         : texte brut issu de charger_document()
    taille_chunk  : nombre de caractères par chunk (défaut 500)
    chevauchement : caractères partagés entre chunks adjacents (défaut 100)

    Retourne
    --------
    List[str] — liste de chunks non vides
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=taille_chunk,
        chunk_overlap=chevauchement,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(texte)
    return [c.strip() for c in chunks if len(c.strip()) > 30]


print("✅ decouper_chunks() définie.")

liste_chunks = decouper_chunks(texte_brut)

print(f"✂️  Nombre de chunks : {len(liste_chunks)}")
print(f"📏 Taille moyenne   : {sum(len(c) for c in liste_chunks) // len(liste_chunks)} caractères")
print("\n--- 3 premiers chunks ---")
for i, chunk in enumerate(liste_chunks[:3], 1):
    print(f"\n[Chunk {i}]\n{chunk}")

def creer_base_vectorielle(chunks: List[str]) -> dict:
    """
    Encode les chunks en vecteurs et construit un index FAISS.

    Paramètres
    ----------
    chunks : liste de chaînes issue de decouper_chunks()

    Retourne
    --------
    dict :
        - "index"  : faiss.IndexFlatL2
        - "chunks" : List[str]
        - "dim"    : int
    """
    if not chunks:
        raise ValueError("La liste de chunks est vide — aucun document à indexer.")

    model = _get_embed_model()
    print(f"⏳ Encodage de {len(chunks)} chunks…")
    vecteurs = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    vecteurs = vecteurs.astype(np.float32)

    dim = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dim)# creer faiss
    index.add(vecteurs)

    print(f"✅ Index FAISS créé : {index.ntotal} vecteurs, dimension {dim}")
    return {"index": index, "chunks": chunks, "dim": dim}


print("✅ creer_base_vectorielle() définie.")

base_vectorielle = creer_base_vectorielle(liste_chunks)

print(f"\n📊 Résumé de la base vectorielle :")
print(f"   • Vecteurs indexés : {base_vectorielle['index'].ntotal}")
print(f"   • Dimension        : {base_vectorielle['dim']}")
print(f"   • Chunks stockés   : {len(base_vectorielle['chunks'])}")

def rechercher(#trouve les passage les plus pertinant pour la question
    question: str,
    base_vectorielle: dict,
    top_k: int = 4,
) -> List[Tuple[str, float]]:
    """
    Trouve les passages les plus pertinents pour une question donnée.

    Paramètres
    ----------
    question         : question posée par l'étudiant (texte libre)
    base_vectorielle : dict retourné par creer_base_vectorielle()
    top_k            : nombre de passages à retourner (défaut 4)

    Retourne
    --------
    List[Tuple[str, float]]
        Liste de (chunk, score_distance) — score faible = très pertinent
    """
    model = _get_embed_model()
    vecteur_q = model.encode([question], convert_to_numpy=True).astype(np.float32)

    index  = base_vectorielle["index"]
    chunks = base_vectorielle["chunks"]

    k_reel = min(top_k, len(chunks))
    distances, indices = index.search(vecteur_q, k_reel)

    resultats = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            resultats.append((chunks[idx], float(dist)))

    return resultats


print("✅ rechercher() définie.")

MA_QUESTION = "Qu'est-ce que la photosynthèse ?"   # ← modifie ici

resultats = rechercher(MA_QUESTION, base_vectorielle, top_k=3)

print(f"🔍 Question : « {MA_QUESTION} »")
print(f"   {len(resultats)} passages trouvés\n")

for i, (chunk, score) in enumerate(resultats, 1):
    print(f"{'─'*60}")
    print(f"  Résultat #{i}  |  Score L2 = {score:.4f}")
    print(f"{'─'*60}")
    print(chunk[:300] + ("…" if len(chunk) > 300 else ""))
    print()

def pipeline_complet(chemin_fichier: str, question: str, top_k: int = 4):
    """
    Exécute les 4 étapes du pipeline RAG et affiche les résultats.
    Retourne aussi la base vectorielle pour réutilisation.
    """
    print("=" * 60)
    print("ÉTAPE 1 — Chargement du document")
    print("=" * 60)
    texte = charger_document(chemin_fichier)
    print(f"✅ {len(texte)} caractères extraits.\n")

    print("=" * 60)
    print("ÉTAPE 2 — Découpage en chunks")
    print("=" * 60)
    chunks = decouper_chunks(texte)
    print(f"✅ {len(chunks)} chunks créés.\n")

    print("=" * 60)
    print("ÉTAPE 3 — Création de la base vectorielle")
    print("=" * 60)
    base = creer_base_vectorielle(chunks)
    print()

    print("=" * 60)
    print("ÉTAPE 4 — Recherche")
    print("=" * 60)
    resultats = rechercher(question, base, top_k=top_k)
    print(f"🔍 Question : « {question} »\n")
    for i, (chunk, score) in enumerate(resultats, 1):
        print(f"[{i}] Score L2 = {score:.4f}")
        print(chunk[:250] + ("…" if len(chunk) > 250 else ""))
        print()

    return base


# ─── Lance le pipeline ───────────────────────────────────────────────────────
# base = pipeline_complet("mon_fichier.pdf", "Quelle est la définition de X ?", top_k=3)
print("✅ pipeline_complet() définie. Décommente la dernière ligne pour lancer.")

# Génère rag_pipeline.py depuis ce notebook
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--to", "script", "rag_pipeline.ipynb",
     "--output", "rag_pipeline"],
    check=True
)
print("✅ rag_pipeline.py exporté dans le répertoire courant.")

