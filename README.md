# NLP-FINE_TUNE-MFCC-API

 the system takes a person's typed text and voice recording, figures out their emotion from each separately, then hands both off to a fusion engine that combines them with the face model's output. Your job was text + speech + the fusion/severity/explanation logic that ties everything together.
Text pipeline — backend/ml/text/
Think of this as an assembly line a piece of text walks through:

code_switch.py — first stop. If someone writes "aaj bahut udaas hoon" (mixed Hindi-English), this swaps recognized Hindi emotion words ("udaas" → "sad") into English before anything else sees the text, because the classifier only understands English.
emotion_model.py — the actual classifier. Takes the (possibly translated) text and outputs probabilities across 7 emotions (joy, sadness, anger, fear, disgust, surprise, neutral) using a pretrained DistilBERT model. If the model can't load (no internet, no GPU), it falls back to a "mock" mode that still gives a stable, believable answer so the rest of the team isn't blocked.
sarcasm_detector.py — a correction layer. Catches things like "great, just great" or "not exactly thrilled" and nudges the probabilities away from a naive positive reading toward something more realistic.
crisis_detector.py — runs independently, in parallel. This isn't one of the 7 emotions — it's a separate yes/no flag for distress/self-harm language, because that signal needs to never get "averaged away" by the other emotions.
inference.py — the foreman. Calls all four in the right order and packages the final result into the JSON shape your README already specified.

train_bert.py is just the optional script to fine-tune your own model later if the pretrained one isn't accurate enough.
Speech pipeline — backend/ml/speech/
Same idea, for audio:

feature_extraction.py turns a .wav file into numbers: MFCCs (a standard way to represent what is being said acoustically) plus pitch and loudness (which capture how it's being said — the tone).
model.py is the actual neural network (CNN + LSTM) that takes those numbers and predicts an emotion.
train_speech.py trains that network on the RAVDESS dataset your README mentions.
inference.py is the one your Flask routes actually call — it loads the trained model if it exists, and if not, makes a reasonable guess based on the raw energy/pitch (loud + shaky voice → leans angry/fearful; quiet + flat → leans sad) so demos still work without a trained model.

Fusion & explanation — backend/app/services/
This is where face + text + speech results get combined:

fusion_engine.py — weighted-averages the three signals (text counts most, since it's the clearest channel), but adjusts those weights per-request: a confident modality counts more, a low-confidence one counts less, and if sarcasm was flagged, it shifts some of text's weight over to speech (tone is a better sarcasm detector than the words themselves).
severity_grader.py — turns the fused result into an action tier (low/moderate/high/critical). Crucially, if the crisis flag from text fired, this overrides everything else — even if the face looked calm.
xai_explainer.py — builds the human-readable "why" behind the result, e.g. "Text signals distress (word 'hopeless') while face shows sadness markers."
