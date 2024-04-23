from pydub import AudioSegment
from pydub.playback import play
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

audio = AudioSegment.from_file("pediatrics_1.mp3", format="mp3")
first_sentence = audio[484.689 * 1000: 490.266 * 1000]
# play(first_sentence)
first_sentence.export("pediatrics_1_sentence_1.wav", format="wav", bitrate="128k")

# wave = wavfile.read("pediatrics_1_sentence_1.wav")
wave = librosa.load("pediatrics_1_sentence_1.wav", sr=16000)


processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

input_features = processor(wave[0], sampling_rate=wave[1], return_tensors="pt").input_features

predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription[0])
