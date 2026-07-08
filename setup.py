from setuptools import setup, find_packages

setup(
    name="scenesense",
    version="0.1.1",
    description="3D spatial scene monitoring — detect missing, moved, and new objects from video or live camera feeds",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Shreyash Saripalli",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "ultralytics>=8.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)