# Spoken language identification (GRN Hack 2025)

A tool that listens to a short audio clip and predicts which language is being spoken. Built during Hack 2025 for Global Recordings Network (GRN), a non-profit that records and archives speech in thousands of languages, many of them rare or endangered.

This repository holds my part of the project: a working desktop demo that records from your microphone and names the language, plus an experiment in adapting the underlying model to GRN's own recordings. The rest of the team built a full web app around the same idea.

## What's in here

There are two separate pieces, and they're worth keeping apart because they use the model differently.

The demo app (`language-detection-app.ipynb`) records a few seconds of speech and runs it straight through Meta's mms-lid-4017 model to get a language. This is the runnable, end-to-end part. It uses the base model exactly as it ships, with no fine-tuning.

The fine-tuning experiment (`fineTune_mms_lid_4017.ipynb`) is research, not a finished product. It tries to specialise mms-lid-4017 for GRN's languages. Full fine-tuning turned out to need far more memory than we had, so I trained a small classifier on top of the frozen model instead. More on that below.

## The model

Both pieces sit on top of `facebook/mms-lid-4017`, part of Meta's Massively Multilingual Speech project. It's a Wav2Vec2 model trained to identify around 4,000 languages from audio. Every language it knows is tagged with an ISO 639-3 code, for example `kor` for Korean or `tsz` for Purépecha.

## Running the demo

The demo was written and run on Windows with Python 3.13, and you'll need a microphone.

Install the dependencies:

```
pip install torch torchaudio transformers soundfile pyaudio keyboard
```

Open `language-detection-app.ipynb` and run the cells in order. The flow is:

1. Record. A cell opens your microphone and keeps recording until you press the `s` key, then saves the audio to `recording.wav`.
2. Load the model. The first run downloads mms-lid-4017 from Hugging Face, which is several gigabytes, so give it a moment.
3. Predict. The recording is resampled to 16 kHz, passed through the model, and the highest-scoring language is printed as an ISO code.

Two things that tend to trip people up on Windows: PyAudio can be awkward to install, and the `keyboard` library sometimes needs elevated permissions to catch the keypress. If the recording cell just hangs, that is usually the reason.

## How the fine-tuning experiment works

The aim was to make the model better at the specific languages in GRN's archive. The short version: we couldn't fine-tune the real model, so I built a lightweight stand-in and tested how far it could go.

Here is the pipeline from start to finish.

Data. GRN's language list lives in a small SQLite database (the "5fish" database). I pulled the first 100 languages and their ISO codes out of it, then downloaded one sample recording per language from GRN's public audio library. Each clip was converted to a mono, 16 kHz, float32 array, which is the format the model expects.

The wall we hit. Fine-tuning mms-lid-4017 properly means updating all of its weights, and that needs roughly 370 GB of RAM and over seven hours per pass through the data. Colab Pro topped out around 170 GB, so this was a non-starter on the hardware we had. Two early attempts at full fine-tuning are left in the notebook, and both failed for this reason.

The workaround. Rather than retrain the model, I froze it and used it as a feature extractor. Each clip goes through the frozen model once, and I take the internal representation it produces, averaged across time, as a fixed-length fingerprint of that clip. Then I train a single small layer, a logistic-regression classifier, to map those fingerprints to language labels. This is cheap: the large model never changes, and only the tiny classifier learns.

Precomputing the fingerprints for 100 clips took about 54 minutes on a Colab TPU. Training the classifier itself took seconds. Over 100 passes through the data, its training loss fell from about 4.8 to near zero.

## Results

On clips the classifier had already seen during training, it sharpened the model's confidence. For the Purépecha (`tsz`) sample, the trained classifier put almost all of its probability on the correct language.

On a clip it had never seen, the picture is more mixed. For an out-of-sample Korean recording, the base model scored `kor` at 0.9995 and the classifier scored it at 0.9988, a tiny drop rather than a gain. On other unseen clips the classifier moved its guesses around but did not reliably land on the right language.

None of this is surprising given the setup. With only one recording per language, the classifier has almost nothing to generalise from. It can memorise the clips it trained on, but it hasn't heard enough of any language to recognise a new voice speaking it. It also only knows the 100 languages it was trained on, so it can't stand in for the base model's full coverage.

The takeaway is narrow but real: training a light classifier on top of a frozen speech model is a cheap and workable way to specialise it, and it clearly responds to new data. To make it beat the base model on unseen audio, it would need far more data per language.

## Where this could go

The natural next step lives inside the web app. When a user records something and the app names the wrong language, they can correct it by picking the right one. That hands you a new labelled recording for free. Collect enough of those and you have the dataset the classifier is missing: many recordings per language, across different voices and settings. A couple of thousand clips per language would be a sensible target. With that data and enough compute, the base model could be fine-tuned properly, which would do better than the classifier shortcut.

## Repository contents

```
language-detection-app.ipynb          The runnable microphone demo
fineTune_mms_lid_4017.ipynb           The fine-tuning experiment
Handover Report - Nadun - Hack.docx   Notes for whoever picks this up next
df_isoLang.csv                        GRN's language list with ISO codes
```

You will also need sample audio to test against. The notebooks pull clips directly from GRN's public library by language ID, so no local dataset is required to reproduce the demo.

## Credits

Built at Hack 2025, hosted by Global Recordings Network in Australia.

- Michael Madry, web app functionality
- Timothy Wildig, backend engineering
- Sarah Lee, UI and UX design
- Nadun, model fine-tuning and language identification (this repository)

The mms-lid-4017 model is from Meta AI's Massively Multilingual Speech project.
