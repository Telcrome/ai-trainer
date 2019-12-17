import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ai-trainer",
    version="0.0.1",
    author="Raphael Schaefer",
    author_email="raphaelschaefer1@outlook.com",
    description="AI Trainer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.rwth-aachen.de/medical_us_ai/annotator",
    packages=setuptools.find_packages(where='./trainer/'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points='''
        [console_scripts]
        ai=trainer.tools.tools:ai
    '''
)