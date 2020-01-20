import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai-trainer",
    version="0.0.4",
    author="Raphael Schaefer",
    author_email="raphaelschaefer1@outlook.com",
    description="AI Trainer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Telcrome/ai-trainer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    download_url='https://github.com/Telcrome/ai-trainer/archive/0.0.3.tar.gz',
    entry_points='''
        [console_scripts]
        trainer=trainer.tools.cli:trainer
    ''',
    install_requires=[
        'numpy',
        'opencv-python',
        'pydicom',
        'PySimpleGui',
        'tensorflow'
    ],
)