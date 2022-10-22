import argparse
# import ipdb
import os
import warnings
from typing import Generator, List, Optional, Tuple, Union, TYPE_CHECKING
from time import perf_counter
import librosa

import numpy as np
import torch
import tqdm

from .audio import pad_or_trim, log_mel_spectrogram, load_audio
from . import audio as aud
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt


if TYPE_CHECKING:
    from .model import Whisper


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    wl=30,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    all_tokens = None,
    all_segments = None,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    print("Transcribing")
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    print("Analysing audio file")
    # import ipdb; ipdb.set_trace()
    if isinstance(audio, str):
        audio = load_audio(audio)
    print(f"Trimming to {wl} seconds")
    audio = audio[:int(wl * aud.SAMPLE_RATE)]
    mel = log_mel_spectrogram(audio)
    print(f"{audio.shape}")

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            segment = pad_or_trim(mel, aud.N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")
    else:
        print(decode_options["language"])

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seek = 0
    input_stride = exact_div(
        aud.N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    # import ipdb; ipdb.set_trace()
    time_precision = (
        input_stride * aud.HOP_LENGTH / aud.SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    if all_tokens is None:
        all_tokens = []
    if all_segments is None:
        all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")
        # import ipdb; ipdb.set_trace()
        return text

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    print("About to enter loop")

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            # import ipdb; ipdb.set_trace()
            print(f"{seek=}")
            start_timer = perf_counter()
            timestamp_offset = float(seek * aud.HOP_LENGTH / aud.SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], aud.N_FRAMES).to(model.device).to(dtype)
            segment_duration = segment.shape[-1] * aud.HOP_LENGTH / aud.SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            # TC: Where the money happens 
            result: DecodingResult = decode_with_fallback(segment)
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            # import ipdb; ipdb.set_trace()
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    end_timer = perf_counter()
                    print(f"{wl=}")
                    print(f"loop duration: {end_timer - start_timer:0.1f}")
                    yield (
                        add_segment(
                            start=timestamp_offset + start_timestamp_position * time_precision,
                            end=timestamp_offset + end_timestamp_position * time_precision,
                            text_tokens=sliced_tokens[1:-1],
                            result=result,
                        ),
                        all_tokens,
                        all_segments,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                end_timer = perf_counter()
                print(f"{wl=}")
                print(f"loop duration: {end_timer - start_timer:0.1f}")
                yield (
                    add_segment(
                        start=timestamp_offset,
                        end=timestamp_offset + duration,
                        text_tokens=tokens,
                        result=result,
                    ),
                    all_tokens,
                    all_segments,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek


    # return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language)


def transcribe_stream(
    model: "Whisper",
    filename = 'Example.ogg',
    **kwargs,
):
    print(filename)
    sr = librosa.get_samplerate(filename)
    print(sr)
    # aud.SAMPLE_RATE = sr
    # aud.HOP_LENGTH = sr // 100
    # # CHUNK_LENGTH = 15 
    # # CHUNK_LENGTH = 30
    # aud.N_SAMPLES = aud.CHUNK_LENGTH * aud.SAMPLE_RATE  # 480000: number of samples in a chunk
    # aud.N_FRAMES = exact_div(aud.N_SAMPLES, aud.HOP_LENGTH)
    # yield from transcribe(model, filename, **kwargs)

    audio_arr = aud.load_audio(filename)
    
    # stream = librosa.stream(
    #     filename,
    #     block_length=aud.N_FRAMES,
    #     frame_length=aud.HOP_LENGTH,
    #     hop_length=aud.HOP_LENGTH
    # )
    FRAMES_PER_SEC = 16000
    BLOCK_SIZE = 5 * FRAMES_PER_SEC
    i = 0
    while i * BLOCK_SIZE < len(audio_arr):
        print(f"block {i}:")
        tokens = []
        segments = []
        for text, tokens, segments in transcribe(model, audio_arr[i * BLOCK_SIZE : (i+1) * BLOCK_SIZE], all_tokens=tokens, all_segments=segments, **kwargs):
            yield text
        # yield from transcribe(
        #     model,
        #     audio_arr[i * BLOCK_SIZE : (i+1) * BLOCK_SIZE],
        #     **kwargs
        # )
        i += 1


def cli():
    from . import available_models

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", choices=available_models(), help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead.")
        args["language"] = "en"

    temperature = args.pop("temperature")
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    from . import load_model
    model = load_model(model_name, device=device, download_root=model_dir)

    for audio_path in args.pop("audio"):
        result = transcribe(model, audio_path, temperature=temperature, **args)

        audio_basename = os.path.basename(audio_path)

        # save TXT
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
            write_txt(result["segments"], file=txt)

        # save VTT
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as vtt:
            write_vtt(result["segments"], file=vtt)

        # save SRT
        with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)


if __name__ == '__main__':
    cli()
