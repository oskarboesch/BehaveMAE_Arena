import os
import sys

import torch

# Make local package imports work when pytest is launched without PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.general_hiera import GeneralizedHiera
from models.hbehave_mae import HBehaveMAE


def test_hbehavemae_simple_verbose_run():
	"""Very small smoke test with verbose prints for all major MAE stages."""
	torch.manual_seed(0)

	device = torch.device("cpu")

	model = HBehaveMAE(
		input_size=(400, 1, 72),
		in_chans=1,
		embed_dim=16,
		num_heads=2,
		out_embed_dims=(16, 32),
		stages=(1, 1),
		q_strides=[(1, 1, 1)],
		mask_unit_attn=(True, False),
		patch_kernel=(1, 1, 1),
		patch_stride=(1, 1, 1),
		patch_padding=(0, 0, 0),
		decoder_embed_dim=32,
		decoder_depth=1,
		decoder_num_heads=2,
		decoding_strategy="single",
		norm_loss=False,
	).to(device)
	model.train()

	x = torch.randn(2, 400, 1, 72, device=device)

	print("\n========== HBehaveMAE verbose smoke test ==========")
	print("Input x shape [B, T, Num Individuals, keypoints * keypoints dim]:", tuple(x.shape))
	print("patch_stride:", model.patch_stride)
	print("q_strides:", model.q_strides)
	print("tokens_spatial_shape:", model.tokens_spatial_shape)
	print("mask_spatial_shape:", model.mask_spatial_shape)
	print("tokens_spatial_shape_final:", model.tokens_spatial_shape_final)
	print("mask_unit_spatial_shape_final:", model.mask_unit_spatial_shape_final)

	x_ch = x.unsqueeze(1)
	print("x with channel [B, C, T, Num Individuals, keypoints * keypoints dim]:", tuple(x_ch.shape))

	with torch.no_grad():
		rand_mask = model.get_random_mask(x_ch, mask_ratio=0.0)
		print("random MU mask shape:", tuple(rand_mask.shape))
		print(
			"keep ratio (empirical):",
			float(rand_mask.float().mean().item()),
		)

		enc_out, stage_intermediates = GeneralizedHiera.forward(
			model,
			x_ch,
			mask=rand_mask,
			return_intermediates=True,
		)
		print("raw encoder output with mask shape:", tuple(enc_out.shape))
		print("num stage intermediates:", len(stage_intermediates))
		for i, feat in enumerate(stage_intermediates):
			print(f"intermediate[{i}] shape:", tuple(feat.shape))

	latent, mask = model.forward_encoder(x_ch, mask_ratio=0.0)
	print("latent (after forward_encoder) shape:", tuple(latent.shape))
	print("mask (forward_encoder) shape:", tuple(mask.shape))
	print("masked ratio (expected around 0.5):", float((~mask).float().mean().item()))

	pred, pred_mask = model.forward_decoder(latent, mask)
	print("pred (decoder output) shape:", tuple(pred.shape))
	print("pred_mask shape:", tuple(pred_mask.shape))
	print("num masked prediction tokens:", int(pred_mask.sum().item()))

	loss, pred_masked, _, _ = model.forward_loss(x_ch, pred, ~pred_mask)
	print("loss:", float(loss.item()))
	print("pred_masked shape:", tuple(pred_masked.shape))

	loss_e2e, pred_e2e, _, _, mu_mask_e2e = model(x, mask_ratio=0.5)
	print("end-to-end loss:", float(loss_e2e.item()))
	print("end-to-end pred shape:", tuple(pred_e2e.shape))
	print("end-to-end MU mask shape:", tuple(mu_mask_e2e.shape))
	print("===================================================\n")

	assert latent.ndim == 6
	assert pred.ndim == 3
	assert pred_mask.ndim == 2
	# assert torch.isfinite(loss)
	# assert torch.isfinite(loss_e2e)
