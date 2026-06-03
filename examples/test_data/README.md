# Local test data folder

Put development audio files here:

```text
examples/test_data/audio_inputs/
```

Supported early inputs include `.wav`, `.webm`, `.mp3`, `.mp4`, `.m4a`, `.ogg`, `.flac`, `.aac`, `.aiff`, `.mov`.

A file may have a misleading extension, for example WebM/Opus content saved as `.wav`. VSLP uses `ffprobe` and `ffmpeg`, so it reads the actual container/codec rather than trusting the suffix.

Optional v1 demographics file:

```text
examples/test_data/metadata/demographics.csv
```

Generated patient/test audio should not be committed to Git. The `audio_inputs` folder is ignored except for `.gitkeep`.
