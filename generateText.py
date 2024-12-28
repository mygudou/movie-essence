import os
import librosa
import torch
import subprocess
import tkinter as tk
from tkinter import filedialog
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 定义全局变量
MODEL_NAME = "openai/whisper-large-v3-turbo"
OUTPUT_DIRECTORY = "./transcriptions"
AUDIO_OUTPUT_DIRECTORY = "./audio_files"

# 加载模型和处理器
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# 确保输出目录存在
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(AUDIO_OUTPUT_DIRECTORY, exist_ok=True)

def video_to_audio(video_path, audio_output_path):
    """将视频文件转换为音频文件"""
    command = [
        "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", audio_output_path
    ]
    subprocess.run(command, check=True)

def transcribe_audio(audio_path):
    """转录音频文件"""
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    if sampling_rate != 16000:
        raise ValueError(f"音频采样率 ({sampling_rate}) 与模型要求的采样率不一致 (16000)。请重新采样。")

    # 定义切片长度（每片 30 秒）
    slice_duration = 30  # 单位：秒
    slice_length = int(slice_duration * sampling_rate)  # 对应样本点数量

    # 将长音频切片
    slices = [audio_array[i: i + slice_length] for i in range(0, len(audio_array), slice_length)]

    # 转录每个切片
    transcriptions = []
    for idx, audio_slice in enumerate(slices):
        print(f"正在处理音频的第 {idx + 1}/{len(slices)} 段...")
        inputs = processor(audio_slice, sampling_rate=16000, return_tensors="pt", padding=True)
        input_features = inputs.input_features

        # 手动创建 attention_mask
        attention_mask = torch.ones_like(input_features).bool()

        # 模型推理
        with torch.no_grad():
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids
            )

        # 解码并存储结果
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        transcriptions.append(transcription)

    # 拼接所有切片的转录结果
    final_transcription = " ".join(transcriptions)
    return final_transcription

def main():
    # 弹出文件选择框
    root = tk.Tk()
    root.withdraw()
    video_files = filedialog.askopenfilenames(title="选择视频文件", filetypes=[("所有支持的视频文件", "*.mp4 *.avi *.mov *.mkv")])

    if not video_files:
        print("未选择任何文件。")
        return

    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"文件不存在: {video_file}")
            continue

        # 提取文件名
        base_name = os.path.splitext(os.path.basename(video_file))[0]

        # 转换为音频文件
        audio_output_path = os.path.join(AUDIO_OUTPUT_DIRECTORY, f"{base_name}.wav")
        print(f"正在将视频文件转换为音频: {video_file} -> {audio_output_path}")
        video_to_audio(video_file, audio_output_path)

        # 转录音频文件
        print(f"正在转录音频文件: {audio_output_path}")
        transcription = transcribe_audio(audio_output_path)

        # 保存转录结果
        transcription_output_path = os.path.join(OUTPUT_DIRECTORY, f"{base_name}.txt")
        with open(transcription_output_path, "w", encoding="utf-8") as f:
            f.write(transcription)

        print(f"转录完成，结果已保存到: {transcription_output_path}")

if __name__ == "__main__":
    main()
