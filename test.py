import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 加载模型和处理器
model_name = "openai/whisper-large-v3-turbo"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# 加载音频文件
audio_file_path = "/Users/gufeng/dwhelper/2.mp3"
audio_array, sampling_rate = librosa.load(audio_file_path, sr=16000)

# 确保采样率一致
if sampling_rate != 16000:
    raise ValueError(f"音频采样率 ({sampling_rate}) 与模型要求的采样率不一致 (16000)。请重新采样。")

# 定义切片长度（每片 30 秒）
slice_duration = 30  # 单位：秒
slice_length = int(slice_duration * sampling_rate)  # 对应样本点数量

# 将长音频切片
slices = [audio_array[i : i + slice_length] for i in range(0, len(audio_array), slice_length)]

# 转录每个切片
transcriptions = []
for idx, audio_slice in enumerate(slices):
    print(f"正在处理第 {idx + 1}/{len(slices)} 段音频...")
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

# 输出结果到文件
output_file_path = "/Users/gufeng/dwhelper/transcription.txt"
with open(output_file_path, "w", encoding="utf-8") as f:
    f.write(final_transcription)

print(f"转录结果已保存到文件: {output_file_path}")
