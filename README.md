# Diff_Retrieval
The dependencies for the environment are the same as for Covr
```
python -m pip install pytorch_lightning --upgrade
python -m pip install hydra-core --upgrade
python -m pip install lightning
python -m pip install einops
python -m pip install pandas
python -m pip install opencv-python
python -m pip install timm
python -m pip install fairscale
python -m pip install tabulate
python -m pip install transformers
```
use opencv == 4.8.0.74

For the correct functioning of the code with the BLIP2 model, you need to install the dev version of LAVIS

```
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```
The code requires Python 3.8 and PyTorch >= 2.0
