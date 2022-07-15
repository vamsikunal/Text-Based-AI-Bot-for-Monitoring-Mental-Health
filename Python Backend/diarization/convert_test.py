import speech_recognition as sr
import librosa
import soundfile as sf

filename = 'audio/audio13.wav'
y, s = librosa.load(filename, sr=22500)
print('Audio Time Series:', y)
print('Sampling Rate:', s)
sf.write('audio/example1_new.wav', y, s)
recognizer = sr.Recognizer()
file = open(r"test.txt", "w")

try:
    with sr.AudioFile('audio/example1_new.wav') as source:
        # listen for the data (load audio to memory)
        audio_data = recognizer.record(source)
        # recognize (convert from speech to text)
        text = recognizer.recognize_google(audio_data)
        print(text, '\n')
        file.writelines(text)
        print('Speech-to-Text File created!')
        file.close()

except sr.RequestError as e:
    print("Could not request results; {0}".format(e))

except sr.UnknownValueError:
    print("unknown error occurred")
