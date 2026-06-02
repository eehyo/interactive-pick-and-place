# XAI506 Interactive Pick and Place

This package contains the final-project code for **XAI506-2026 Intelligent
Robot**. The project builds an interactive pick-and-place pipeline in MuJoCo:
students choose a target object from a small tabletop scene, the system obtains
an egocentric RGB-D observation, segments the selected object and the placement
region, converts the masks into 3D targets, and executes a dual-arm robot
pick-and-place rollout through inverse kinematics.

The main notebook is:

```text
notebook/10_interactive_pick_and_place.ipynb
```

The notebook is intentionally explicit: it keeps the simulation loop visible
while the package provides reusable scene, perception, IK, Qt, and rendering
helpers. This makes the code suitable both as a runnable demo and as a scaffold
for assignment blanks.

## Project Flow

At a high level, the interactive pipeline is:

1. Build a MuJoCo scene from robot, table, floor, object, and bowl MJCF assets.
2. Reset and settle the scene inside the notebook-controlled runtime loop.
3. Select a target object using a Qt radio-button widget.
4. Capture an egocentric RGB-D image from the robot head camera.
5. Segment the selected object and placement region.
6. Convert segmentation masks and depth into world-frame point clouds.
7. Solve upper-body IK targets for approach, grasp, lift, place, release, and retreat.
8. Render simulation, point clouds, IK targets, and egocentric RGB overlays.

## Structure

```text
package/
├── api/            # Notebook-facing objects, perception, IK, and pick-place API
├── hf_models/      # Minimal Hugging Face/SAM2 helpers
├── init_scripts/   # Jupyter/IPython and Qt setup helpers
├── kinematics/     # Transform utilities
├── mujoco_sim/     # MuJoCo parser, viewer, IK, and MJCF helpers
├── qt/             # Qt widgets used by the interactive selector
├── utility/        # Timers, visualization, sampling, and small utilities
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

From the repository root:

```bash
conda create -n interactive_pick_place python=3.10 pip
conda activate interactive_pick_place
cd package
pip install -r requirements.txt
pip install -e .
```

Because this is an editable install, changes to `.py` files under `package/`
are reflected without reinstalling.

For notebook use, register the conda environment as a kernel if needed:

```bash
python -m ipykernel install --user --name interactive-pick-place
```

## Notebook Usage

The current final-project notebook uses package-level wildcard imports to keep
the notebook cells compact:

```python
from package.init_scripts.init_ipython_setup import *
from package.init_scripts.init_qt import *
from package.api import *
from package.api.pick_and_place2 import *
from package.hf_models import *
from package.qt import *
from package.utility import *
```

The central runtime class for the final-project notebook is
`InteractivePickAndPlaceLoopController` in `package/api/pick_and_place.py`.
It owns scene reset, settling, target selection, perception setup, and execution
while still returning explicit `step` and `render` dictionaries to the notebook
loop.

## Perception Backend

The default notebook can use Gemini Robotics-ER with SAM2, but the assignment
API is backend-agnostic. A custom segmenter only needs to return a dictionary
with a binary mask:

```python
{
    "mask": mask,  # np.ndarray, shape (H, W), dtype bool or integer
}
```

This allows students to replace the perception component with manual clicks,
SAM-only segmentation, a saved mask, or another vision model without changing
the IK and simulation parts.

## External Assets

The notebook still expects external assets outside this package:

- Objaverse-style object models, e.g. `asset/objaverse`
- MuJoCo robot/table/floor assets under `asset/`
- local SAM2 checkpoint directory, e.g. `../hf_models/sam2`

Adjust those paths in the notebook or API call arguments for your workspace.
