# tng-whisper
Innovation Hacking, whisper

## Goal
Use Whisper, an NLP tool, to transcribe live text.

### Sub-group #1
Tim Crundall and Bilel Ghorbel

We investigated the whisper transcribe function which is built for converting entire audio files in one go.
We attempted to modify it to handle chunks of audio arrays.

Challenges:
- Whisper models seems to have hard-coded audio chunks of minimum 30 seconds
- Differing sampling rates have to be considered when loading different files

### Sub-group #2
Moritz Maus and Abraham Gutierrez

Goal:
convert measured audio input into python readable format

