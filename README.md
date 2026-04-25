# 📚 Research Paper RAG - Moteur de Recherche Intelligent

Un système **Retrieval-Augmented Generation (RAG)** qui permet de poser des questions en français sur des articles scientifiques et des documentations techniques, et obtenir des réponses intelligentes avec sources.

## ✨ Fonctionnalités

- 🔍 **Recherche sémantique** sur multiple PDFs (articles + docs techniques)
- 🤖 **Génération de réponses** intelligentes basées sur le contenu
- 📖 **Citations automatiques** des sources utilisées
- 🌍 **Support multilingue** (français, anglais)
- 💾 **Stockage vectoriel local** (Chroma)
- 🚀 **Interface utilisateur** simple avec Streamlit
- ⚡ **Optimisé pour PC faible** (modèles légers, local-first)

## 📋 Architecture

```
PDFs → Extraction texte → Embeddings → Vector DB (Chroma)
                                             ↓
                                    Recherche sémantique
                                             ↓
                                    Contexte + Question
                                             ↓
                                       Génération réponse
                                             ↓
                                    Réponse + Sources
```

## 🚀 Démarrage rapide

### 1. Prérequis
- Python 3.8+
- 2-4 GB RAM disponible
- ~6.5 GB d'espace disque

### 2. Installation

```bash
# Cloner le repo
git clone <ton-repo-url>
cd research-paper-rag

# Créer virtual environment
python -m venv venv

# Activer venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Installer dépendances
pip install -r requirements.txt
```

### 3. Préparer les données

1. Télécharge 10-15 PDFs:
   - Articles arXiv (https://arxiv.org)
   - Documentations techniques (PyTorch, TensorFlow, etc.)

2. Place-les dans les dossiers:
   ```
   data/documents/scientific_papers/  → Articles scientifiques
   data/documents/technical_docs/     → Documentations techniques
   ```

### 4. Lancer l'app

```bash
# Préprocess les documents (une seule fois)
python -m src.preprocess

# Lancer Streamlit
streamlit run frontend/streamlit_app.py

# Ouvre http://localhost:8501
```

## 📦 Structure du projet

```
research-paper-rag/
├── src/
│   ├── __init__.py
│   ├── embedder.py           # Création d'embeddings
│   ├── vector_store.py       # Gestion Chroma
│   ├── rag_pipeline.py       # Pipeline RAG complet
│   ├── pdf_parser.py         # Extraction PDF
│   └── utils.py              # Fonctions utiles
├── api/
│   └── main.py               # API FastAPI (optionnel)
├── frontend/
│   └── streamlit_app.py      # Interface utilisateur
├── data/
│   ├── documents/
│   │   ├── scientific_papers/
│   │   └── technical_docs/
│   └── chroma_db/            # Base vectorielle (créée auto)
├── notebooks/
│   └── exploration.ipynb     # Notebooks d'exploration
├── tests/
│   └── test_rag.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## 🔧 Configuration

Crée un fichier `.env` basé sur `.env.example`:

```bash
cp .env.example .env
```

Édite `.env` si tu veux changer les paramètres.

## 📖 Utilisation

### Interface Streamlit (recommandée)

```bash
streamlit run frontend/streamlit_app.py
```

Puis:
1. Tape ta question en français
2. Clique "Chercher et répondre"
3. Vois la réponse + sources

### API REST (optionnel, après Étape 14)

```bash
uvicorn api.main:app --reload
```

Puis requête:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "Comment fonctionne l'\''attention ?"}'
```

## 🛠️ Tech Stack

| Composant | Tech |
|-----------|------|
| Embeddings | SentenceTransformers |
| Vector DB | Chroma |
| LLM | Ollama / HuggingFace API |
| API | FastAPI |
| Frontend | Streamlit |
| Data | Pandas, PyPDF |

## 📊 Performance

| Métrique | Valeur |
|----------|--------|
| Modèle embedding | all-MiniLM-L6-v2 (22 MB) |
| RAM usage | ~2-3 GB |
| Temps recherche | ~100 ms |
| Temps réponse total | ~5-30 sec (dépend LLM) |

## ⚙️ Phases de développement

- [x] Phase 1 : Setup & structure
- [ ] Phase 2 : Core RAG (Étapes 4-7)
- [ ] Phase 3 : Génération (Étapes 8-10)
- [ ] Phase 4 : Interface (Étapes 11-13)
- [ ] Phase 5 : Déploiement (Étapes 14-15)
- [ ] Phase 6+ : Features bonus (Analytics, feedback, etc.)

## 🤝 Contribuer

Des suggestions ? Issues ? PRs bienvenues !

## 📝 Licence

MIT License - voir LICENSE

## 👤 Auteur

Créé par Ony RANDRIAMBOLOLONA - Portfolio Project

---

**Dernière mise à jour** : [Date]
**Version** : 0.1.0