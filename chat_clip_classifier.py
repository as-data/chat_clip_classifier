#!/usr/bin/env python
# coding: utf-8

# # 🐱 Chat Clip Classifier

# In[1]:


# 📦 Installation des dépendances
get_ipython().system('pip install opencv-python moviepy torch torchvision ftfy regex tqdm')
get_ipython().system('pip install git+https://github.com/openai/CLIP.git')


# In[5]:


# 📦 Imports nécessaires
import os
import torch
import clip
from PIL import Image
import cv2
import subprocess


# In[23]:


# 🏷️ Définition des comportements du chat (labels CLIP)
labels = [
    "a cat playing with a toy",
    "a cat sleeping curled up",
    "a cat eating from a bowl",
    "a cat grooming itself",
    "a cat walking around",
    "a cat jumping",
    "a cat looking at the camera",
    "a cat hiding under furniture",
    "a cat running fast",
    "a cat sitting still",
    "a cat meowing with its mouth open",
    "a cat interacting affectionately with a human"
]
for label in labels:
    folder = f"clips_output/{label.replace(' ', '_')}"
    os.makedirs(folder, exist_ok=True)


# In[25]:


# 🧠 Chargement du modèle CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(labels).to(device)


# In[27]:


# 🎞️ Extraction des frames d’une vidéo
def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)
    count = 0
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{img_count}.jpg")
            cv2.imwrite(frame_path, frame)
            img_count += 1
        count += 1
    cap.release()


# In[29]:


# 🏷️ Classification d’une frame avec CLIP
def classify_frame(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    return labels[probs.argmax()], probs.max()


# In[31]:


# ✂️ Découpe d’un extrait vidéo avec ffmpeg
def extract_clip(video_path, start_time, end_time, output_path):
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c", "copy", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# In[33]:


# 🧩 Traitement complet d'une vidéo (extraction + classification + découpe)
def process_video(video_path):
    basename = os.path.basename(video_path).split('.')[0]
    frames_folder = f"extracted_frames/{basename}"
    extract_frames(video_path, frames_folder)

    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        print(f"⚠️ Aucune frame extraite pour {basename}, vidéo ignorée.")
        return

    timestamps = []
    for i, filename in enumerate(frame_files):
        label, prob = classify_frame(os.path.join(frames_folder, filename))
        timestamps.append((i, label))

    grouped = []
    current_label = timestamps[0][1]
    start = timestamps[0][0]
    for i in range(1, len(timestamps)):
        if timestamps[i][1] != current_label:
            grouped.append((current_label, start, timestamps[i-1][0]))
            current_label = timestamps[i][1]
            start = timestamps[i][0]
    grouped.append((current_label, start, timestamps[-1][0]))

    for idx, (label, start_frame, end_frame) in enumerate(grouped):
        duration = end_frame - start_frame + 1  # approx en secondes si frame_rate=1

        if duration < 2:
            print(f"⏩ Clip ignoré ({label}) – durée trop courte : {duration} sec")
            continue  # on passe au suivant

        output_name = f"clips_output/{label.replace(' ', '_')}/{basename}_{label.replace(' ', '_')}_{idx}.mp4"
        extract_clip(video_path, start_frame, end_frame + 1, output_name)



# In[35]:


# 🔁 Traitement automatique de toutes les vidéos .mp4 et .mov
for video in os.listdir("input_videos"):
    if video.lower().endswith((".mp4", ".mov")):
        print(f"▶️ Traitement de : {video}")
        process_video(f"input_videos/{video}")


# In[1]:


import os
import cv2

input_dir = "input_videos"
output_dir = "clips_output"

# 📦 1. Nombre de vidéos brutes
raw_videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".mov"))]
nb_raw_videos = len(raw_videos)

# 📁 2. Parcours des clips générés
total_clips = 0
total_duration_sec = 0
clips_per_category = {}

for category in os.listdir(output_dir):
    cat_path = os.path.join(output_dir, category)
    if os.path.isdir(cat_path):
        clips = [f for f in os.listdir(cat_path) if f.endswith(".mp4")]
        clips_per_category[category] = len(clips)
        total_clips += len(clips)
        
        # Calculer la durée totale
        for clip in clips:
            clip_path = os.path.join(cat_path, clip)
            cap = cv2.VideoCapture(clip_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            total_duration_sec += duration
            cap.release()

# ⏱️ 3. Moyenne durée clip
avg_duration = total_duration_sec / total_clips if total_clips > 0 else 0

# 📊 Résumé
print("📊 Statistiques du traitement vidéo :\n")
print(f"🎥 Vidéos brutes traitées : {nb_raw_videos}")
print(f"🎬 Clips générés : {total_clips}")
print(f"⏱️ Durée moyenne par clip : {avg_duration:.1f} sec")
print("🗂️ Répartition par catégorie :")
for cat, count in clips_per_category.items():
    print(f"   - {cat} : {count} clips")

