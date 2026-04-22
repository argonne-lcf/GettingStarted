# ParaView Tutorial: Interactive Visualization and Automation with Python

Welcome to this tutorial on using **ParaView** for scientific visualization on ALCF systems.
By the end of this webinar, you will be able to explore data interactively on ALCF filesystems,
save and reuse visualization states, capture and edit Python scripts, and leverage Python for
automation and fine-grained control.

---

## Table of Contents

1. [Setup for Interactive Exploration on ALCF Filesystems](#1-setup-for-interactive-exploration-on-alcf-filesystems)
2. [Saving ParaView State for Future Use](#2-saving-paraview-state-for-future-use)
3. [Capturing Interactions with Python](#3-capturing-interactions-with-python)
4. [Automating a Camera Rotation Around Your Data](#4-automating-a-camera-rotation-around-your-data)
5. [Using Python for Finer Control of ParaView](#5-using-python-for-finer-control-of-paraview)

---

## 1. Setup for Interactive Exploration on ALCF Filesystems

This section walks through how to set up an interactive ParaView session on a Polaris compute node.
The same steps apply to all other ALCF compute resources.

### Step 1: Launch an Interactive Session

Start by requesting an interactive session on a compute node through the job scheduler.
Once your session is granted, you will land on a compute node (e.g., `x3006c0s13b0n0`).

### Step 2: Load the ParaView Module

First, make the ALCF visualization modules available, then search for the ParaView module:

```bash
module use /soft/modulefiles/
module spider paraview
```

This will list the available ParaView versions:

```
------------------------------------------------------------------------------------------------------------------------------
  visualization/paraview:
------------------------------------------------------------------------------------------------------------------------------
     Versions:
        visualization/paraview/paraview-5.12.0-EGL
        visualization/paraview/paraview-5.12.1-EGL
        visualization/paraview/paraview-5.13.1-EGL
        visualization/paraview/paraview-6.0.1-EGL

------------------------------------------------------------------------------------------------------------------------------
  For detailed information about a specific "visualization/paraview" package (including how to
  load the modules) use the module's full name.
  Note that names that have a trailing (E) are extensions provided by other modules.
  For example:

     $ module spider visualization/paraview/paraview-6.0.1-EGL
------------------------------------------------------------------------------------------------------------------------------
```

Load the desired version (here we use the latest, `6.0.1-EGL`):

```bash
module load visualization/paraview/paraview-6.0.1-EGL
```

> **Note:** The `-EGL` suffix indicates that this build uses EGL for offscreen rendering,
> which is required on headless compute nodes without a display.

### Step 3: Launch the ParaView Server (`pvserver`)

Start `pvserver` on the compute node, specifying a port and the available display indices:

```bash
pvserver --server-port=8000 --displays=0,1,2,3 $PARAVIEW_PVSERVER_OPTS
```

If successful, you will see:

```
Waiting for client...
Connection URL: cs://x3006c0s13b0n0:8000
Accepting connection(s): x3006c0s13b0n0:8000
```

Leave this terminal open — `pvserver` will keep running and wait for your client to connect.

### Step 4: Create an SSH Tunnel

On your **local machine**, open a new terminal and create an SSH tunnel that forwards your
local port `8000` to the `pvserver` port on the compute node via the Polaris login node:

```bash
ssh -v -N -L 8000:x3006c0s13b0n0:8000 polaris.alcf.anl.gov
```

- `-N` — do not execute a remote command; just forward the port
- `-L 8000:x3006c0s13b0n0:8000` — forward local port `8000` to port `8000` on the compute node

> **Tip:** Replace `x3006c0s13b0n0` with the hostname of the compute node assigned to your
> interactive session.

### Step 5: Connect the ParaView Client

Open the **ParaView GUI** on your local machine and connect to the server through the tunnel:

1. Go to **File → Connect**
2. Click **Add Server**
3. Set the host to `localhost` and port to `8000`
4. Click **Connect**

ParaView will now render data on the compute node and stream the results to your local client,
giving you full interactive access to data on any ALCF filesystem.

[↑ Back to Table of Contents](#table-of-contents)

---

## 2. Saving ParaView State for Future Use

ParaView allows you to save your entire visualization pipeline as a **state file** so you can
reload it in a future session without repeating your steps. In this section, we build a simple
pipeline and save it as a Python state file.

### Step 1: Create a Basic Pipeline

We will use two built-in ParaView tools to build a quick example pipeline:

1. **Add a Wavelet source** — Go to **Sources → Wavelet**. This generates a synthetic dataset
and is a convenient built-in source for testing and demonstration purposes.
2. **Apply an Isosurface filter** — With the Wavelet selected, go to **Filters → Isosurface**
(or search for "Contour" in the filter search bar). Apply the filter to extract a surface
at a chosen scalar value.
3. Click **Apply** to render the isosurface in the viewport.

You should now have a simple two-step pipeline: `Wavelet → Contour`.

### Step 2: Save State as a Python File

Once your pipeline is ready, save the state:

1. Go to **File → Save State**
2. In the save dialog, change the file type to **Python State File (`.py`)**
3. Make sure to select **"Use File Names Relative to"** and choose **local paths** — this
ensures the state file is portable and references files correctly on your local system
4. Choose a filename and save

> **Important:** Selecting "Python" (rather than the default `.pvsm` XML format) gives you a
> human-readable script that you can edit, reuse, and build upon.

### Step 3: Inspect the Python Script

Open the saved `.py` file in any text editor. You will see that ParaView has translated every
step of your pipeline into `paraview.simple` API calls — the source creation, filter
application, display properties, colormap settings, and camera position are all captured.

This script is a great starting point for learning the ParaView Python API and for automating
your visualizations, which we will build on in the sections that follow.

[↑ Back to Table of Contents](#table-of-contents)

---

## 3. Capturing Interactions with Python

ParaView has a built-in **Python tracing** feature that records every action you perform in
the GUI as equivalent Python API calls. This is one of the best ways to learn the ParaView
Python API — simply interact with the GUI and let ParaView write the code for you.

### Step 1: Reset the Pipeline

Before starting a fresh trace, clear the current pipeline:

1. Select all sources in the **Pipeline Browser**
2. Press **Delete** to remove them, or go to **Edit → Delete All**

This gives you a clean slate so that the captured trace only reflects the interactions you
intend to record.

### Step 2: Load the State Saved in the Previous Step

Rather than rebuilding the pipeline manually, reload the Python state file you saved in
Section 2:

1. Go to **File → Load State**
2. Select your previously saved `.py` state file
3. Confirm the file paths when prompted

Your `Wavelet → Contour` pipeline will be restored exactly as you left it.

### Step 3: Enable Python Trace Capture

With the pipeline loaded, start recording your interactions:

1. Go to **Tools → Start Trace**
2. A dialog will appear with trace options — the defaults are generally fine
3. Click **OK** to begin recording

From this point on, every interaction you perform in the GUI — adjusting filter parameters,
changing colormaps, moving the camera, toggling visibility — will be captured as Python code.

Perform a few interactions, then stop the trace:

1. Go to **Tools → Stop Trace**

### Step 4: Inspect the Captured Code

Once you stop the trace, ParaView will automatically open a **Python Script Editor** window
containing the generated code. You will see that your GUI interactions have been translated
into `paraview.simple` API calls line by line.

You can:

- **Save** the script for later use via **File → Save** in the script editor
- **Edit** the script to generalize or extend it
- **Run** the script directly from the editor to replay your interactions

This captured script forms the foundation for the automation and scripting techniques covered
in the sections that follow.

[↑ Back to Table of Contents](#table-of-contents)

---

## 4. Automating a Camera Rotation Around Your Data

ParaView's **Animation** system lets you automate a full 360° camera orbit around your data
with just a few clicks — no scripting required. This is useful for generating flyaround
animations to showcase your visualization.

### Step 1: Open the Time Manager

Go to **View → Animation Panel** (or **View → Time Manager**, depending on your ParaView
version). This opens the animation controls at the bottom of the screen.

### Step 2: Add a Camera Follow Path

1. In the Animation Panel, click the large **`+`** button to add a new animation track
2. From the dropdown, select **Camera** and choose **Follow Path**
3. A `Camera-RenderView1` track will appear in the animation timeline

### Step 3: Set the Number of Frames

Adjust the total number of frames for the animation. **100 frames** is a good starting point
for quick results — enough for a smooth rotation without being too slow to generate.

### Step 4: Play the Animation

Press the **Play** button. The camera will perform a full **360° orbit** around your data,
aligned with the world axes (Y points up). You can save this animation via
**File → Save Animation** to export it as an image sequence or video file.

---

### Customizing the Camera Orbit

The default orbit is axis-aligned, but you can define a custom orbit from any camera angle.

#### Adjust the Camera to a New View

Use the mouse to rotate, pan, and zoom the viewport until the camera is in your desired
position and orientation.

#### Delete the Old Camera Track and Add a New One

If you press Play at this point, ParaView will execute the **original** orbit and ignore your
new camera position. To fix this:

1. Select the existing `Camera-RenderView1` track in the Animation Panel and delete it
2. Click the **blue `+`** button to add a new Camera track

#### Define the Orbit from the Current View

1. **Double-click** on the white band of the new `Camera-RenderView1` track — this opens the
**Animation Keyframes** editor
2. Verify that the camera is in your desired position; adjust in the viewport if needed
3. Click **Create Orbit** — a dialog will appear asking for orbit parameters
4. Click **OK** on the **"Create Orbit"** dialog, then **OK** on the **Animation Keyframes**
dialog to confirm

Press **Play** again. The camera will now perform a full 360° orbit around your data from
your custom viewpoint.

[↑ Back to Table of Contents](#table-of-contents)

---

## 5. Using Python for Finer Control of ParaView

The ParaView GUI covers most common visualization tasks, but the built-in **Python Shell**
gives you direct access to the full `paraview.simple` API for finer control. A practical
workflow is to use an **LLM** (such as ChatGPT, Claude, or Gemini) to generate boilerplate
code snippets, which you can then paste and run directly in ParaView's Python Shell.

### Step 1: Open the Python Shell

Go to **View → Python Shell**. This opens an interactive Python console at the bottom of the
ParaView window where you can type or paste code and execute it immediately.

---

### Step 2: Query the Current Camera State

A common first task is to inspect the current camera position so you can use it as a
reference in scripts. Ask any LLM:

> *"Give me ParaView 6.0 Python code to print the current camera position, focal point, and view up vector."*

The result will look something like this:

```python
from paraview.simple import *

view = GetActiveViewOrCreate('RenderView')
camera = view.GetActiveCamera()

print("Position:", camera.GetPosition())
print("Focal Point:", camera.GetFocalPoint())
print("View Up:", camera.GetViewUp())
```

Paste this into the Python Shell and press **Enter** to print the current camera state.
You can use these values as inputs for more advanced scripts.

---

### Step 3: Create a Camera Orbit Animation via Python

The **Create Orbit** widget in the GUI (covered in Section 4) always aligns the orbit to the
world axes, which can be limiting if your data is panned or viewed from an oblique angle.
Python gives you full control over the orbit path.

Ask any LLM:

> *"Give me ParaView 6.0 Python code to create a 360° camera orbit animation around the current
> camera view, respecting the current position and focal point."*

```python
from paraview.simple import *
import math

view = GetActiveViewOrCreate('RenderView')
camera = view.GetActiveCamera()

# Get current camera state
position = camera.GetPosition()
focal = camera.GetFocalPoint()
viewup = camera.GetViewUp()

# Set up animation scene
scene = GetAnimationScene()
scene.NumberOfFrames = 100

# Get or create the camera track
cue = GetCameraTrack(view=view)

# Build keyframes rotating 360 degrees around focal point
keyframes = []
num_frames = 100

for i in range(num_frames):
    camera.Azimuth(360.0 / num_frames)
    kf = CameraKeyFrame()
    kf.KeyTime = i / float(num_frames)
    kf.Position = camera.GetPosition()
    kf.FocalPoint = camera.GetFocalPoint()
    kf.ViewUp = camera.GetViewUp()
    kf.ViewAngle = camera.GetViewAngle()
    keyframes.append(kf)

cue.KeyFrames = keyframes
```

> **Advantage over the GUI:** This approach builds the orbit from the **current camera
> position and focal point**, so it works correctly even when the view is panned or tilted —
> overcoming a limitation of the built-in **Create Orbit** widget.

---

### Step 4: Save Each Frame to Create a Movie

Once your camera animation is set up, you can export every frame as an image and assemble
them into a video. Ask any LLM:

> *"Give me ParaView 6.0 Python code to save each frame of the current animation as a PNG file. I have 100 frames and my resolutionn is 1920x1080."*

```python
from paraview.simple import *

# Saves frame_0000.png, frame_0001.png, ... frame_0099.png
SaveAnimation(
    '/path/to/output/frame.png',   # base filename — ParaView appends 4-digit frame number
    GetActiveView(),
    ImageResolution=[1920, 1080],     # width x height in pixels
    FrameRate=24,
    FrameWindow=[0, 99]            # frame range to save (0 to NumberOfFrames-1)
)
```

Replace `/path/to/output/` with your desired output directory. Once the frames are saved,
you can stitch them into a video using `ffmpeg`:

```bash
module use /soft/modulefiles/; module load spack-pe-base ffmpeg
ffmpeg -r 25 -i /path/frame.%04d.png -r 25 -pix_fmt yuv420p rotation.mp4
```

[↑ Back to Table of Contents](#table-of-contents)

