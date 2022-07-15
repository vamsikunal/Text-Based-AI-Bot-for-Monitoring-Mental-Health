def stt():
    import os
    import glob
    from os import listdir
    import librosa
    import speech_recognition as sr
    import re
    from scipy.io import wavfile
    import numpy as np
    import soundfile as sf

    path = 'speakers/'
    text_files = [f for f in os.listdir(path) if f.endswith('.wav')]
    text_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(text_files)
    print('Files count:', len(text_files))
    recognizer = sr.Recognizer()

    def convert(filename, output, speakerid):
        file = open(output, "a")
        print(filename)
        # y, s = librosa.load(filename, sr=8000)
        # sf.write(filename, y, s)
        try:
            with sr.AudioFile(filename) as source:
                # listen for the data (load audio to memory)
                audio_data = recognizer.record(source)
                # recognize (convert from speech to text)
                # text = speakerid + ': '
                text = recognizer.recognize_google(audio_data)
                print(speakerid + ': ' + text)
                file.writelines(speakerid + ': ' + text + '\n')
                file.close()

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            # print("unknown error occurred")
            pass

    i = 0
    for filename in text_files:
        str1 = path + filename
        str2 = path + 'speech_transcript.txt'
        if i % 2 == 0:
            convert(str1, str2, 'speaker0')
        else:
            convert(str1, str2, 'speaker1')
        i += 1
    print('Speech-to-Text File created!')
