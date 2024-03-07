# MS1Former

# Installation
```
git clone https://github.com/sanomics-lab/MS1Former.git
cd MS1Former
conda env create -f environment.yaml
source activate mspectra
```
# Data preprocessing

```
python dataset.py --file_dir ./mzml --save_dir ./mzml_parsed
```
# Train and Test

```
python train_cnn.py
python predict.py
```
