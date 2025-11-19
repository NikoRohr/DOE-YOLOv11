import kagglehub
import os

# current file path
proj_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(proj_dir, "..", "data", "foggy_cityscape")

# need to specify the output directory as envrionment variable
# export KAGGLEHUB_CACHE=/path/to/your/preferred/directory
os.environ["KAGGLEHUB_CACHE"] = output_dir

# Download latest version
path = kagglehub.dataset_download("khitdon/foggy-cityscapes-yolo")

print("Path to dataset files:", path)
