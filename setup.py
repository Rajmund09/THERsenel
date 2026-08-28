from setuptools import setup, find_packages

setup(
    name="thermal-border-intrusion",
    version="1.0.0",
    description="Automated Border Intrusion Detection Using Thermal-Visible Image Fusion and YOLOv8",
    author="Thermal Border Security Team",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tensorboard>=2.15.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
        "scikit-learn>=1.4.0",
        "onnx>=1.15.0",
        "onnxruntime>=1.17.0",
        "pillow>=10.2.0",
        "albumentations>=1.4.0",
    ],
)
