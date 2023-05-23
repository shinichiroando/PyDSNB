# Python calculator of the diffuse supernova neutrino background (PyDSNB)

The codes allow to calculate the spectrum of the diffuse supernova neutrino background (DSNB). They take various models of supernova neutrino spectrum and star-formation and supernova generation rates in the Unvierse. The MSW neutrino flavor oscillations are fully taken into account for both the normal and inverted mass hierarchy.

## Authors

- Shin'ichiro Ando
- Nick Ekanger
- Shunsaku Horiuchi

We have checked that the codes work with python 3.9 but cannot guarantee for other versions of python. In any case, we cannot help with any technical issues not directly related to the content of PyDSNB (such as installation, sub-packages required, etc.)

## Setup

Installing SNEWPY package is essential. It can be downloaded and installed from https://github.com/SNEWS2/snewpy. We checked that PyDSNB works with SNEWPY v1.3.

Data directory 'SNEWPY_models' must also be downloaded from the same repository and put in the same directory as the PyDNSB.py main file.

## How to use PyDSNB

The file 'PyDSNB.py' constains all the essential functions that are used to compute various quantities relevant to DSNB, such as star-formation rate and supernova neutrino spectrum. Please also read 'PyDSNB.ipynb' The main class 'DSNB' takes an initialization file. We include 'params.ini' as an example. The content of this file should be self-explanatory.
