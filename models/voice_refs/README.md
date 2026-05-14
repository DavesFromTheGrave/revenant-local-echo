# Voice References

Drop reference audio clips of the target voice here. Revenant Echo uses
[Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) to clone
the voice from a reference WAV with zero training time.

## Requirements

- **Format:** WAV, mono, 16-bit PCM
- **Sample rate:** 24 kHz is ideal; Chatterbox will resample anything
- **Duration:** **must be longer than 5 seconds.** 10–30 s is the sweet spot
- **Content:** clean, single speaker, no background music or other voices
- **Quality:** dialog from a game cutscene, podcast episode, or audiobook
  works well. Phone-quality voicemail does not.

## How to make one

If you have a video clip with the voice you want, extract and convert it
with ffmpeg:

```powershell
# extract audio, mono, 44.1kHz from a video
ffmpeg -i "source.mp4" -vn -acodec pcm_s16le -ar 44100 -ac 1 "ref.wav"

# concat multiple clips for more variety (better cloning)
@"
file 'C:/path/to/clip1.wav'
file 'C:/path/to/clip2.wav'
"@ | Set-Content clips.txt -Encoding ascii

ffmpeg -f concat -safe 0 -i clips.txt -acodec pcm_s16le -ar 24000 -ac 1 "combined.wav"
```

Save the result as `models/voice_refs/<name>.wav` and point at it from
`config.yaml`:

```yaml
tts:
  voice_ref: "models/voice_refs/your_voice.wav"
```

## Caching

When V starts up, Chatterbox loads the reference ONCE and caches the
voice conditionals in memory. Every subsequent sentence skips the
conditioning step, so per-sentence synthesis is fast (~1.5 s on an
RTX 3070 Ti for short sentences). Swapping the reference requires
restarting V.

## Licensing note

If the voice you're cloning is copyrighted (game character, celebrity,
etc.), the reference WAV stays on your machine. **Don't commit it,
don't redistribute it.** This folder is gitignored except for this
README for exactly that reason.
