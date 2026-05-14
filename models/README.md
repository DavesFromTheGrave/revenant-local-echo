# Models

This directory holds the wake-word `.onnx` files Revenant Echo loads at startup.
The model binaries themselves are **not** committed — fetch them yourself before
running V.

## Default wake word: "Hey Friday"

Revenant Echo ships configured to use the community-trained "Hey Friday" model
from the [Home Assistant Wake Words Collection](https://github.com/fwartner/home-assistant-wakewords-collection).

### Automatic install

`scripts/install.ps1` downloads this file for you. If you ran that, you're done.

### Manual install

Download the model and save it as `models/hey_friday.onnx`:

```powershell
$url = "https://github.com/fwartner/home-assistant-wakewords-collection/raw/main/en/hey_friday/hey_Friday%21.onnx"
Invoke-WebRequest -Uri $url -OutFile "models\hey_friday.onnx"
```

The path in `config/config.yaml` (`wake_word.model: "models/hey_friday.onnx"`)
is project-relative and resolves to this directory automatically.

## Using a different wake word

`config/config.yaml` accepts either:

- **A built-in OpenWakeWord model name** — `"alexa"`, `"hey_jarvis"`,
  `"hey_mycroft"`, `"hey_rhasspy"`. These auto-download into OpenWakeWord's
  package directory on first run.
- **A path to a custom `.onnx` file** — relative to the project root
  (e.g. `"models/your_word.onnx"`) or absolute.

There are ~100 community-trained wake words at
<https://github.com/fwartner/home-assistant-wakewords-collection/tree/main/en>.
Download any `.onnx` file, drop it in this folder, and point
`wake_word.model` at it.

## Training a custom wake word

To train your own phrase ("Wake up V", "Hey Echo", whatever), use the
[OpenWakeWord training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)
on Google Colab. It generates synthetic training data and trains an ONNX
model in ~30-60 minutes on a free Colab GPU. Drop the resulting `.onnx`
file into this directory.
