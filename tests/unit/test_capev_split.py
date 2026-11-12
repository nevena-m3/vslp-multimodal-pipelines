"""258_SENT_CAPEV: six sentences in one file are cut in the pauses between them."""
from __future__ import annotations

import wave

import numpy as np

from vslp.acoustic.alignment import batch
from vslp.acoustic.alignment.capev_split import (SENTENCES, align_words, plan_split,
                                                 write_slice)


def _timed(texts_and_times):
    words = []
    for text, start, end in texts_and_times:
        parts = text.split()
        step = (end - start) / len(parts)
        words += [{"word": word, "start": start + i * step, "end": start + (i + 1) * step}
                  for i, word in enumerate(parts)]
    return words


def _energy(duration, speech, hop=0.01):
    energy = np.full(int(duration / hop), -80.0)
    for start, end in speech:
        energy[int(start / hop):int(end / hop)] = -20.0
    return energy


def test_misheard_words_still_align_in_order():
    said = "the blue spot is on a key again how hard did he hit em".split()
    target = "the blue spot is on the key again how hard did he hit him".split()
    matched = align_words(said, target)
    assert matched[0] == 0 and matched[target.index("hit")] == said.index("hit")
    assert matched[target.index("key")] == said.index("key")
    assert sum(index is not None for index in matched) >= 12


def test_cuts_fall_in_the_pauses_between_the_six_sentences():
    # Dysarthric-style transcript, an examiner comment and uneven pauses.
    spoken = [("okay go ahead", 0.2, 0.9), ("the blue spot is on a key again", 1.5, 3.4),
              ("how hard did he hit him", 4.4, 5.8), ("we were away a year ago", 6.3, 7.9),
              ("we eat eggs every easter", 9.5, 11.0), ("my mama makes lemon jam", 11.6, 13.0),
              ("peter will keep at the peak", 14.2, 15.9)]
    words = _timed(spoken)
    energy = _energy(17.0, [(a, b) for _t, a, b in spoken])
    spans = plan_split(words, energy, 0.01, 17.0)
    assert [span["status"] for span in spans] == ["OK"] * 6
    pauses = [(3.4, 4.4), (5.8, 6.3), (7.9, 9.5), (11.0, 11.6), (13.0, 14.2)]
    for span, (a, b) in zip(spans, pauses):
        assert a <= span["end"] <= b
    assert 0.9 < spans[0]["start"] < 1.5  # the examiner comment is left out
    assert spans[-1]["end"] <= 17.0


def test_missing_sentence_and_no_pause_are_flagged_not_cut():
    spoken = [("the blue spot is on the key again", 1.0, 3.0),
              ("how hard did he hit him", 3.5, 5.0),
              ("we eat eggs every easter", 6.0, 7.5),
              ("my mama makes lemon jam peter will keep at the peak", 8.0, 12.0)]
    words = _timed(spoken)
    spans = {span["sentence"]: span for span in
             plan_split(words, _energy(13.0, [(1, 3), (3.5, 5), (6, 7.5), (8, 12)]), 0.01, 13.0)}
    assert spans["S5"]["status"] == "SENTENCE_NOT_FOUND"
    assert spans["S6"]["status"] == spans["S9"]["status"] == "NO_PAUSE_BETWEEN_SENTENCES"
    assert spans["S3"]["status"] == "OK"


def test_sentence_wavs_are_exact_copies_and_map_to_their_prompts(tmp_path):
    rate = 44100
    samples = (np.random.default_rng(0).standard_normal(rate * 3) * 3000).astype(np.int16)
    source = tmp_path / "CNES01_1075_1_20240101_258_SENT_CAPEV.wav"
    with wave.open(str(source), "wb") as handle:
        handle.setnchannels(1), handle.setsampwidth(2), handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    target = tmp_path / "out" / f"{source.stem}_HIT.wav"
    write_slice(source, target, 1.0, 2.5)
    with wave.open(str(target), "rb") as handle:
        assert handle.getframerate() == rate
        copied = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    assert np.array_equal(copied, samples[rate:int(2.5 * rate)])
    names = [f"{source.stem}_{suffix}.wav" for suffix, _task, _text in SENTENCES]
    inventory = batch.build_inventory(None, names)
    assert inventory.batch_role.eq("fixed_prompt").all()
    assert inventory.task_id.tolist() == [task for _s, task, _t in SENTENCES]


def test_pause_is_measured_on_the_audio_not_on_stretched_word_times():
    # CAPE-V test, real files: Whisper ended each last word at the next
    # sentence's start, although the audio had a 0.6 s silence there.
    spoken = [("the blue spot is on the key again", 1.0, 3.6), ("how hard did he hit him", 3.6, 5.6),
              ("we were away a year ago", 5.6, 7.6), ("we eat eggs every easter", 7.6, 9.6),
              ("my mama makes lemon jam", 9.6, 11.6), ("peter will keep at the peak", 11.6, 13.6)]
    real_speech = [(1.0, 3.0), (3.6, 5.0), (5.6, 7.0), (7.6, 9.0), (9.6, 11.0), (11.6, 13.0)]
    spans = plan_split(_timed(spoken), _energy(14.5, real_speech), 0.01, 14.5)
    assert [span["status"] for span in spans] == ["OK"] * 6
    for span, (_a, end_of_speech), (next_start, _b) in zip(spans, real_speech, real_speech[1:]):
        assert end_of_speech < span["end"] < next_start
        assert span["pause_after"] >= 0.4
