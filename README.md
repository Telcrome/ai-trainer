# Installation

Open anaconda powershell, navigate into the annotator repo and execute:

```bash
conda env create -f environment.yml
```

## Optional dependencies

Annotator helps with building data generator and it relies on imgaug for it
```bash
conda config --add channels conda-forge
conda install imgaug
```

For handling the dicom format annotator uses pydicom.
```bash
pip install -U pydicom
```

## Usage in other Projects

For usage with conda you can either
```
conda develop .
```
or
```bash
pip install git+https://git.rwth-aachen.de/medical_us_ai/annotator@v0
```
