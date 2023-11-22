# (Conditional) Variational Auto-Encoder for GPS Data

This is a companion repository of the preprint: "Auto-encoding GPS data to reveal individual and collective behaviour".

- Pytorch dictionaries of the models learnt from real data are in the `save` folder
- Synthetic anonymized data generated with the decoder of the model CVAE3 are in the `data` folder

# Training a new model from the data
For a CVAE model with 3 latent dimensions
python train_traj_vae.py --nz 3 --model "cvae"

For a VAE model with 4 latent dimensions
python train_traj_vae.py --nz 4 --model "vae"
