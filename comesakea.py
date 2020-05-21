import speech_recognition as sr
from os import path
audio =  ("kiicomesa.wav")
r = sr.Recognizer()

with sr.AudioFile(audio) as source:
    r.adjust_for_ambient_noise(source)
    audio = r.record(source)
    try:
        text = r.recognize_google(audio, language="pl")
        print("Working on....")
        print(text)

    except Exception as e:
        print(e)
