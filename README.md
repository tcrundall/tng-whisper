# tng-whisper
Innovation Hacking, whisper

## Goal
Use Whisper, an NLP tool, to transcribe live text.

### Sub-group #1
Tim and Bilel

We investigated the whisper transcribe function which is built for converting entire audio files in one go.
We attempted to modify it to handle chunks of audio arrays.

Challenges:
- Whisper models seems to have hard-coded audio chunks of minimum 30 seconds
- Differing sampling rates have to be considered when loading different files

### Sub-group #2
Goal:
convert measured audio input into input for python

