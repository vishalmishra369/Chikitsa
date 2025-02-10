import pyttsx3

def speak(text, accent="en-IN"):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            
                engine.setProperty('voice', voice.id[7])
                break

        engine.setProperty('rate', 150)  # Adjust the speaking rate
        engine.setProperty('pitch', 200)  # Adjust the pitch
        engine.setProperty('volume', 1.0)  # Adjust the volume

        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        print(f"pyttsx3 failed: {e}")


