# 📄 Ask Your PDF


## 📌 Description du projet

Ask Your PDF est une application interactive développée avec Streamlit qui permet de poser des questions en langage naturel à partir d’un fichier PDF. Le projet illustre l’utilisation des LLMs (Large Language Models) et de la recherche sémantique via embeddings pour interroger des documents, tout en mettant l’accent sur :

La recherche sémantique via embeddings et base vectorielle (FAISS)

La segmentation de texte pour gérer de gros documents (text splitters)

La simplicité d’utilisation via une interface web interactive

## ⚙️ Fonctionnalités principales

-📄 Upload PDF : importer un document PDF à interroger.

-🔎 Recherche sémantique : retrouver les passages les plus pertinents via OpenAI embeddings et FAISS.

-💬 Question/Réponse : poser une question et obtenir une réponse contextualisée à partir du contenu du PDF.

-🖥️ Interface Streamlit : interface web simple pour uploader le fichier et saisir les questions.

## 🛠️ Technologies utilisées

Python 3.11
Streamlit (interface web interactive)
LangChain (pipeline LLM + text splitters + QA chain)
FAISS (vector database pour la recherche sémantique)
OpenAI Embeddings (représentation sémantique des textes)
PyPDF2 (extraction du texte des PDF)
python-dotenv (gestion des variables d’environnement)

# 🎯 Objectif pédagogique

Ce projet est conçu à des fins éducatives et démonstratives.
Il illustre les concepts de :

-Recherche sémantique dans un document PDF

-Question/Réponse avec LLMs

-Construction d’une base vectorielle pour interroger de gros textes
