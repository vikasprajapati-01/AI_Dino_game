# AI Dino Game

An experimental reinforcement-learning agent that learns to play the Chrome Dino game using a Deep Q-Network (DQN). The project captures the game window from the screen, converts each frame into a grayscale observation, sends keyboard actions to the game, and rewards the agent for surviving.

## How it works

- **Observation:** A screen capture of the configured game area, resized to a single-channel `83 x 100` grayscale image.
- **Actions:** Jump (`space`), duck (`down`), or do nothing.
- **Game-over detection:** OCR through Tesseract when available, with pixel-sum detection as a fallback.
- **Reward:** `+1` for each surviving step and `-100` when game over is detected.
- **Algorithm:** Stable-Baselines3 DQN with a convolutional policy.
- **Checkpoints:** The training callback saves models under `models/`.

## Requirements

This project is intended to run locally on a desktop with a visible browser window. It needs permission to capture the screen and control the keyboard and mouse.

- Python 3.9 or newer
- Google Chrome or another browser running the Chrome Dino game
- A desktop session with a display
- Windows is recommended because the input automation is implemented with `pydirectinput`
- Tesseract OCR is optional; the program falls back to pixel-based detection when it is unavailable

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies. For a CPU-only PyTorch installation:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install "stable-baselines3[extra]" gymnasium mss pydirectinput pytesseract opencv-python matplotlib "numpy<2.0" "pandas<2.1" protobuf
```

For an NVIDIA CUDA 11.8 installation, replace the PyTorch command with:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

To enable OCR-based game-over detection, install the Tesseract application separately and ensure its executable is available on `PATH`. The Python package alone does not include the Tesseract engine.

## Running training

1. Open the Chrome Dino game in a visible browser window. Navigating to `chrome://dino` is the simplest option.
2. Position the browser so the coordinates in `dino_DQN.py` cover the game and the game-over message.
3. Activate the virtual environment.
4. Run:

```bash
python dino_DQN.py
```

The script creates the environment, starts DQN training, periodically saves checkpoints in `models/`, saves the final model as `dino_dqn_final.zip`, and closes the environment when training finishes.

Keep the mouse and keyboard available while the program is running. `pydirectinput` will click the game and press the selected action keys automatically.

## Configuration

The default capture regions are defined in `ModernWebGame`:

```python
game_location = {'top': 150, 'left': 400, 'width': 600, 'height': 400}
done_location = {'top': 230, 'left': 680, 'width': 500, 'height': 50}
```

Adjust these values if the browser window is positioned differently. The `use_ocr` argument controls whether OCR is attempted when Tesseract is installed. The environment also limits an episode to 1,500 steps before truncation.

The current training entry point uses 1,000 total timesteps. Increase `total_timesteps` in `dino_DQN.py` for a longer training run. Checkpoints are configured for every 25,000 callback steps, so a run shorter than that interval will only produce the final model unless an older checkpoint is already present.

## Included models

- `dino_dqn_final.zip`: Final model artifact committed to the repository.
- `models/model_step_25000.zip`: A saved DQN checkpoint from training step 25,000.

Stable-Baselines3 model archives are binary files and should be loaded with the same compatible Python and library environment used for training.

## Capturing a screenshot

`image.py` is a small utility for checking the configured capture region. It takes a screenshot of the game area and displays it with OpenCV:

```bash
python image.py
```

Use this utility to verify that the coordinates include the game before starting training. Close the preview window to finish the command.

## Project files

```text
dino_DQN.py                 DQN environment, training configuration, and callbacks
image.py                    Screen-region preview utility
dino_dqn_final.zip          Final trained model
models/model_step_25000.zip Saved training checkpoint
```

## Troubleshooting

### The observation is blank or captures the wrong area

Move the browser to match the configured coordinates or update `game_location` and `done_location`. Run `image.py` first to inspect the capture area.

### Keyboard input does not reach the game

Make sure the browser is visible and focused, then allow the Python process to control the keyboard and mouse. Avoid moving the mouse while training is active.

### Tesseract is not found

This is supported. The script prints that it is using pixel-based detection instead. Install Tesseract and add it to `PATH` if OCR detection is preferred.

### Linux display or automation errors

Screen capture and desktop input require an active graphical session. Headless servers and terminal-only sessions are not supported by the current implementation.

## Notes

This is an experimental learning project rather than a packaged application. Screen coordinates, timing, browser zoom, display scaling, and game layout can affect performance. Run it only in a controlled desktop session where automated keyboard and mouse input is safe.