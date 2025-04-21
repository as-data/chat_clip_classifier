# 🐱 Chat Clip Classifier

Ce projet utilise l'IA pour analyser automatiquement des vidéos de chat et extraire les moments clés (jeu, sommeil, repas, miaulement, interaction avec un humain…) sous forme de clips classés.

---

## 🎯 Objectif

Créer un pipeline de traitement vidéo assisté par IA, capable de :
- Détecter les comportements d’un chat dans une vidéo
- Découper automatiquement les séquences correspondantes
- Classer les clips dans des dossiers selon le comportement détecté

---

## 🧠 Technologies utilisées

- [CLIP (OpenAI)](https://github.com/openai/CLIP) pour la classification visuelle, ce modèle permet de relier une image à un concept exprimé en langage naturel
- Python / Jupyter Notebook
- OpenCV pour l’extraction de frames
- ffmpeg pour le découpage vidéo

---

## 📊 Statistiques du traitement (exécution locale)

- 🎥 **Vidéos brutes traitées :** 358
- 🎬 **Clips générés automatiquement :** 1157
- ⏱️ **Durée moyenne par clip :** 3,0 sec
- 🗂️ **Catégories détectées :** 12

| Catégorie                                         | Nb. de clips |
|--------------------------------------------------|--------------|
| 🐟 a_cat_eating_from_a_bowl                      | 57           |
| 🧼 a_cat_grooming_itself                         | 248          |
| 🫣 a_cat_hiding_under_furniture                  | 146          |
| ❤️ a_cat_interacting_affectionately_with_a_human | 160          |
| 🐾 a_cat_jumping                                 | 113          |
| 👁️ a_cat_looking_at_the_camera                  | 44           |
| 🗣️ a_cat_meowing_with_its_mouth_open            | 55           |
| 🧸 a_cat_playing_with_a_toy                      | 151          |
| 🏃 a_cat_running_fast                            | 56           |
| 🪑 a_cat_sitting_still                           | 6            |
| 😴 a_cat_sleeping_curled_up                      | 74           |
| 🚶 a_cat_walking_around                          | 47           |

> 📂 Tous les clips sont organisés automatiquement dans des sous-dossiers de `clips_output/` selon leur catégorie.

---

## 📸 Démonstration : montage final à partir des clips générés

🎬 Voici un exemple de séquence vidéo montée à partir des clips générés automatiquement par le script.
Chaque extrait provient d’un clip détecté et classé selon le comportement du chat (jeu, sommeil, interaction…).
👉 [Voir la vidéo sur TikTok](https://vm.tiktok.com/ZNdY5CoNp/)