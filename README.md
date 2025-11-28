# DOE-YOLOv11

Project Fall 2025 for the course Computational Imaging (CSED520) at POSTECH.

## Setup
Clone git repo with submodules.
```bash
git clone --recursive https://github.com/NikoRohr/DOE-YOLOv11.git
```

Install required python packages.
```bash
# Install requirements
pip install -r requirements.txt

# Uninstallation of previous ultralytics package may be required
# pip uninstall ultralytics

# Install custom ultralytics with pado module
cd ultralytics
pip install -e .
```

Download `foggy_cityscape` dataset by running the script `./scripts/download_foggy_cityscape.py`.
