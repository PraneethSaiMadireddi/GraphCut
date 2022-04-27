# Graphcut Textures: Image and Video Synthesis Using Graph Cuts

## Installation

The implementation is in Python 3.

Install the required Python packages using the following command:
```
pip install -r requirements.txt
```

The `networkx` library is used for graph plotting.

## Running
Command to generate new textures:
  ```
  python3 graphcut.py <image_path> <generated_image_path> <height> <width>
  ```
Example:
  ```
  python3 graphcut.py images/input_images/berry.gif images/generated_images/berry_new.gif 512 512
  ```
