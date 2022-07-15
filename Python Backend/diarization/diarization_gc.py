import io
import argparse
import json

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


def transcribe_gcs(gcs_uri):
    from google.cloud import speech_v1p1beta1 as speech
    client = speech.SpeechClient()

    audio = speech.types.RecognitionAudio(uri=gcs_uri)

    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000,
        language_code='en-US',
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
        model='phone_call')

    operation = client.long_running_recognize(config, audio)
    print("Waiting for operation to complete..")
    response = operation.result(timeout=90)

    wanted_result = response.results[-1]
    # print("WantedResult:" + str(wanted_result))
    # print("Type:"+ str(type(wanted_result)))
    prev_speaker_tag = wanted_result.alternatives[0].words[0].speaker_tag
    # print("PrevSpeakerTag" + str(prev_speaker_tag))
    # print("alternatives" + str(wanted_result.alternatives[0]) +str(type(wanted_result.alternatives[0])))
    # print("Words" + str(wanted_result.alternatives[0].words[0]) + str(type(wanted_result.alternatives[0].words[0])))

    # for result in wanted_result.alternatives[0]:
    #     if(result.words[0].speaker_tag != prev_speaker_tag):
    #         print('/n')
    #         print("Speaker {}:".format(result.words[0].speaker_tag))
    #         l.append(result.words[0].word)

    #     else:
    #         l.append(result.alternatives.words.word)

    #     prev_speaker_tag = result.alternatives.words.speaker_tag

    s = "Speaker {}:".format(prev_speaker_tag)
    for i in wanted_result.alternatives[0].words:
        if i.speaker_tag != prev_speaker_tag:
            # print('\n')
            # print("Speaker {} : ".format(i.speaker_tag))
            print(s + '\n')
            s = "Speaker {}:".format(i.speaker_tag)
        s += " " + i.word
        prev_speaker_tag = i.speaker_tag
    print(s + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('path', help='gcs path for audio file')
    args = parser.parse_args()
    transcribe_gcs(args.path)