# Almond

## Get Started
- Python = 3.8

```bash
pip install -r requirements.txt
```

## Training VAE
Before training the decoder of ALMOND, you must train VAE.
```python
python train_vae.py \
--gpus 4 \
--max_epochs 10 \
--data_name mnist \
--num_workers 4 \
--batch_size 512 \
--latent_dim 10 \
--output_dim 784 \ # When we use MNIST dataset
--hidden_dim 500 300 \ # The hidden dimensions of MLP (output_dim -> hidden_dim -> latent_dim)
--learning_rate 0.0003
```

## Training Almond
After training VAE is done, you can start training Almond with pretrained VAE.

```python
python train_vae.py \
--gpus 4 \
--max_epochs 10 \
--data_name mnist \
--num_workers 4 \
--step_size 0.02 \
--total_step 10000 \
--batch_size 512
```
