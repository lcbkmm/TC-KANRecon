# TC-KANRecon

Overall structure of the TC-KANRecon:

![model](img/model.png)

Dynamic Clipping Strategy Process:

![Strategy](img/Dynamic_clipping.png)

Model Generate Detailed Effect Comparison:

![renderings](img/view.png)

## 1. Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -r requirements .
```

## 2. Data Preparation

The two datasets we used are both public datasets. For firstMRI, you can find it in [Link](https://fastmri.med.nyu.edu/), which includes 1172 subjects with more than 41,020 slice data; for SKM-TEA, you can find it in [Link](https://stanfordaimi.azurewebsites.net/datasets/4aaeafb9-c6e6-4e3c-9188-3aaaf0e0a9e7), which includes 155 subjects with more than 24,800 slice data. Both of them use the single-coil data of their knee.

When you have your data set ready, you need to change your data set path in the configuration file below:

- for vae:
```
python config/vae/config_monaivae_zheer.py
```
- for model:
```
python config/diffusion/config_zheer_controlnet.py
```

## 3. Training
- for vae:

```
python my_vqvae/train_vae.py

```
- for model:

```
python stable_diffusion/train_sd.py
python stable_diffusion/trian_model.py

```

## 4. Evaluating

```
python stable_diffusion/val_model.py

```

## 5. Citation
```
Xiao Yu, Yifei Chen
```
