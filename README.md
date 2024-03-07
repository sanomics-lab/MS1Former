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
python dataset.py --file_dir ./mzml/raw --save_dir ./mzml/parsed/IPX0000937000_resolution_10_sparse
```
# Train and Test

```
python train_cnn.py
python predict.py
```
