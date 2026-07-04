from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import os

proj_root = os.path.dirname(__file__)
plots = [
    os.path.join(proj_root, 'notebooks', 'plots', 'model_accuracy_comparison.png'),
    os.path.join(proj_root, 'notebooks', 'plots', 'univariate_annual_rainfall_dist.png'),
    os.path.join(proj_root, 'notebooks', 'plots', 'heatmap_correlation.png'),
    os.path.join(proj_root, 'notebooks', 'plots', 'univariate_flood_class_counts.png'),
]
plots = [p for p in plots if os.path.exists(p)]

# Short script ~80-90 seconds voiceover trimmed to ~90s by clip durations
script = (
    "This project predicts flood risk using historical rainfall and event data. "
    "It preprocesses rainfall inputs, trains several models, and serves a simple web interface for predictions. "
    "The notebook includes exploratory analysis and model comparisons. "
    "Use the `train_model.py` script to retrain models and `app.py` to run the web demo locally. "
    "See the plots for model accuracy and rainfall distributions. "
    "Visit the repository for code, data generation, and instructions to reproduce the results."
)

# Create narration
tts = gTTS(script, lang='en')
narration_path = os.path.join(proj_root, 'narration.mp3')
tts.save(narration_path)

# Create title slide
W, H = 1920, 1080
title_path = os.path.join(proj_root, 'video_title.png')
img = Image.new('RGB', (W, H), color=(8, 70, 102))
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('arial.ttf', 80)
except Exception:
    font = ImageFont.load_default()
d.text((120, 200), 'Rising Waters — Flood Prediction', font=font, fill=(255, 255, 255))
d.text((120, 320), 'Project demo and highlights', font=font if font else None, fill=(220, 230, 240))
img.save(title_path)

# Build clips list: title + plots
clips = []
clips.append(ImageClip(title_path).resize(width=1920))
for p in plots:
    clips.append(ImageClip(p).resize(width=1920))

# Load audio length and assign durations
audio = AudioFileClip(narration_path)
total_audio = audio.duration
n_clips = len(clips)
duration_per = max(3, total_audio / n_clips)

video_clips = []
start = 0.0
for c in clips:
    clip = c.set_duration(duration_per)
    video_clips.append(clip)

final = concatenate_videoclips(video_clips, method='compose')
final = final.set_audio(audio.set_start(0))
output_path = os.path.join(proj_root, 'demo.mp4')
final.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac', threads=4, preset='medium')
print('Wrote', output_path)
