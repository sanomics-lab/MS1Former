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

Model checkpoint can be downloaded from [zenodo](!https://zenodo.org/api/records/10791588/draft/files/model-ms1former.pt/content).

```
python train_cnn.py
python predict.py
```
