# L3amel Lfahem (العامل الفاهم)
### Enterprise Multilingual RAG & Document Intelligence Platform

Assistant intelligent pour informer, guider et expliquer les droits, obligations et
procédures du **Code du Travail marocain** — en Français, Darija, Arabe classique ou
Anglais — avec une architecture RAG de niveau entreprise (hybrid search, multi-agents,
gouvernance, FinOps).

---

## 1. Aperçu

| | |
|---|---|
| **Stack** | Streamlit · Groq (Llama 3.1/3.3) · FAISS · BM25 · DuckDB · sentence-transformers |
| **Langues** | Français, Darija (code-switching), Arabe classique, Anglais |
| **Déploiement cible** | Streamlit Community Cloud / serveur privé |

## 2. Fonctionnalités (32 capacités réparties en 7 domaines)

| Domaine | Fonctionnalités clés | Fichier |
|---|---|---|
| 1. Ingestion & Extraction | OCR fallback, chunking parent-child, métadonnées auto, PDF diff, détection d'anomalies | `rag_engine.py` |
| 2. Linguistique régionale | Alignement FR/Darija/AR/EN, cross-lingual retrieval, sentiment, traduction des citations | `rag_engine.py`, `agents.py` |
| 3. Recherche haute performance | Hybrid BM25+Dense (fusion RRF), cross-encoder re-ranking, HyDE, décomposition multi-requêtes | `rag_engine.py` |
| 4. Sécurité & gouvernance | Score de confiance, masquage PII, anti-prompt-injection, registre d'audit | `rag_engine.py` |
| 5. Expérience utilisateur | Voice Hub (STT/TTS), exports Word/PDF, auto-dashboarding, Mermaid, quiz | `app.py` |
| 6. FinOps & architecture | Data rooms, tracker tokens/coût, RBAC simulé, boucle RLHF, cascade routing | `app.py`, `agents.py` |
| 7. Agents collaboratifs | Multi-agents (Juridique/Financier/Synthèse), Text-to-SQL DuckDB, conformité réglementaire, fallback local | `agents.py`, `rag_engine.py` |

## 3. Arborescence

```
l3amel_lfahem/
├── .streamlit/
│   └── config.toml       # Thème Enterprise Slate
├── app.py                 # Interface Streamlit (4 workspaces)
├── rag_engine.py           # Moteur RAG, FinOps, audit, exports
├── agents.py                # Orchestration multi-agents, routing, fallback
├── styles.py                 # Design system CSS "Anti-AI look"
├── requirements.txt
└── README.md
```

## 4. Installation locale

```bash
git clone <votre-repo>
cd l3amel_lfahem
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

À l'ouverture, collez votre clé API Groq (obtenue sur [console.groq.com](https://console.groq.com))
dans le champ prévu en sidebar — ou définissez la variable d'environnement `GROQ_API_KEY`
avant de lancer Streamlit.

## 5. Déploiement Streamlit Community Cloud

1. Poussez ce dossier sur un dépôt GitHub public ou privé.
2. Sur [share.streamlit.io](https://share.streamlit.io), créez une nouvelle app pointant
   vers `app.py`.
3. Dans **Secrets**, ajoutez :
   ```toml
   GROQ_API_KEY = "votre_clé"
   ```
4. Déployez. Le fichier `.streamlit/config.toml` applique automatiquement le thème.

## 6. Dépendances système optionnelles

Certaines fonctionnalités avancées dépendent de binaires externes non installables via
pip seul. L'application **dégrade gracieusement** si l'un d'eux est absent (message
explicite affiché à l'utilisateur plutôt qu'un plantage) :

| Fonctionnalité | Dépendance système | Installation (Debian/Ubuntu) |
|---|---|---|
| OCR sur pages scannées | `tesseract-ocr` | `apt-get install tesseract-ocr` |
| Rendu image des pages PDF | `poppler-utils` | `apt-get install poppler-utils` |
| Mode déconnecté / LLM local | [Ollama](https://ollama.com) | binaire séparé, lancé en local |

Sur Streamlit Community Cloud, ajoutez un fichier `packages.txt` à la racine avec :
```
tesseract-ocr
poppler-utils
```

## 7. Notes d'implémentation honnêtes

- **Re-ranking** : utilise un cross-encoder multilingue léger
  (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). S'il ne peut pas être téléchargé
  (pas de réseau sortant), le pipeline continue avec le score de fusion RRF seul.
- **Score de confiance** : dérivé des scores de retrieval (proxy de pertinence
  sémantique), pas d'un modèle de détection d'hallucination entraîné dédié — c'est un
  indicateur d'aide à la décision, pas une garantie absolue d'exactitude.
- **PII masking / anti-injection** : implémentés par expressions régulières et motifs
  heuristiques, volontairement simples et transparents. Pour un usage réellement
  critique, envisager une couche supplémentaire (ex. Presidio, Llama Guard).
- **FinOps** : la table de tarification dans `rag_engine.py` est indicative ; vérifiez
  la tarification Groq en vigueur avant de vous y fier pour de la facturation réelle.
- **RBAC** : simulation applicative (filtrage des data rooms par rôle en session), pas
  un système d'authentification/autorisation de niveau production — à brancher sur un
  fournisseur d'identité réel (SSO/OIDC) avant tout déploiement sensible.
- **Conformité réglementaire** : le module compare un document à des extraits déjà
  indexés du Code du travail ; il ne remplace pas un avis juridique professionnel
  (voir garde-fou intégré au prompt système).

## 8. Avertissement légal

L3amel Lfahem informe sur le contenu du Code du travail marocain à titre pédagogique et
d'orientation générale. Il ne constitue pas un conseil juridique personnalisé et
engageant. Pour toute situation précise, consultez un avocat ou l'inspection du travail
compétente.

## 9. Licence & contact

Projet pédagogique — YaneCode Academy. Adaptez la licence selon l'usage prévu (MIT
recommandé pour un dépôt public).
