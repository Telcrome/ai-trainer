conda env create -f environment.yml
conda activate trainer_env

# Remove the old auto-generated docs
rm -rf ./documentation_source/modules

# Generating docs from code
sphinx-apidoc.exe ./trainer/ -o ./documentation_source/modules

# Compiling the docs
make html