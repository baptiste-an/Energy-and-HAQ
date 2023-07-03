# Healthcare systems' environmental footprints

The goal of this project is to calculate the footprints of healthcare systems for every impact/satellite account available in EXIOBASE. We endogenize capital using data from Södersten and colleagues. We compare the results with the regional Healthcare Access and Quality (HAQ) index.

## Requirements

```bash
$ pip install -r requirements.txt
```

## Usage

Running runme.py requires 40GB of disk space as it will download all exiobase data.

```bash
$ python runme.py
```
It will:

* download EXIOBASE data and save it in Data/EXIO3
* download Kbar data and save it in Data/Kbar, convert files to .feather
* run functions from XXX.py to calculate the footprints
* run functions from XXX.py to 


## Citation

Andrieu, B., Marrauld, L., Vidal, O., Egnell, M., Boyer, L., Fond, G., The exponential relationship between healthcare systems’ resource footprints and their access and quality: a study of 49 regions between 1995 and 2015