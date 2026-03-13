import os
import sys

import pytest
import torch
import math

# Make local package imports work when pytest is launched without PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hbehave_mae import HBehaveMAE
from models.hiera_utils import Reroll, Unroll

def _patch_embed_output_shape(input_size, patch_kernel, patch_stride, patch_padding):
    return [
        math.floor((i + 2*p - k) / s) + 1
        for i, k, s, p in zip(input_size, patch_kernel, patch_stride, patch_padding)
    ]

def _print_header(title: str) -> None:
	print("\n" + "=" * 90)
	print(title)
	print("=" * 90)


def _print_tensor_info(name: str, tensor: torch.Tensor) -> None:
	print(
		f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
		f"device={tensor.device}, min={float(tensor.min()):.6f}, max={float(tensor.max()):.6f}"
	)


# def test_unroll_reroll_roundtrip_verbose_token_sweep():
# 	"""Verbose round-trip test for Unroll -> Reroll with multiple temporal token counts."""
# 	torch.manual_seed(0)

# 	spatial_shape = (1, 72)
# 	patch_kernel = (1, 1, 1)
# 	patch_stride = (1, 1, 1)
# 	patch_padding = (0, 0, 0)
# 	stages = (2, 3)
# 	stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
# 	q_strides = [(2, 1, 2), (2, 1, 2)]
# 	batch_size = 2
# 	embd_dim = 8
# 	q_pool = len(q_strides)

# 	# Compatibility shim for the current implementation in hiera_utils.

# 	temporal_sizes = [64, 128, 400]

# 	_print_header("Unroll/Reroll roundtrip sweep")
# 	print("spatial_shape:", spatial_shape)
# 	print("patch_kernel:", patch_kernel)
# 	print("unroll_schedule:", q_strides)

# 	for temporal_len in temporal_sizes:
# 		input_size = (temporal_len, *spatial_shape)
# 		unroll = Unroll(spatial_shape, patch_stride, q_strides)

# 		print("\n raw input shape (before tokenizer):", (batch_size, temporal_len, *spatial_shape))
# 		patch_embd_output_shape = _patch_embed_output_shape(
#             input_size, patch_kernel, patch_stride, patch_padding
#         )
# 		print(f"patch_embed output shape (tokens): {batch_size} {tuple(patch_embd_output_shape)}")
# 		n_tokens = math.prod(patch_embd_output_shape)
# 		print("calculated n_tokens:", n_tokens)
# 		# Deterministic structured values make debugging easier than random tensors.
# 		x = torch.arange(
# 			batch_size * n_tokens * embd_dim, dtype=torch.float32
# 		).reshape(batch_size, n_tokens, embd_dim)
# 		print("\n--- tokens shape: ", tuple(x.shape))

# 		_print_tensor_info("x_input", x)

# 		x_unrolled = unroll(x)
# 		_print_tensor_info("x_unrolled", x_unrolled)

# 		# Compatibility shim: Reroll expects self.size to exist in current implementation.
# 		reroll = Reroll(
# 			input_size=input_size,
# 			patch_stride=patch_kernel,
# 			unroll_schedule=q_strides,
# 			stage_ends=stage_ends,
# 			q_pool=q_pool,
# 		)

# 		x_rerolled = reroll(x_unrolled, block_idx=0)
# 		_print_tensor_info("x_rerolled_spatial", x_rerolled)

# 		x_restored = x_rerolled.reshape(batch_size, n_tokens, embd_dim)
# 		_print_tensor_info("x_restored_flat", x_restored)

# 		max_abs_err = float((x_restored - x).abs().max().item())
# 		print("max_abs_err(restored,input):", max_abs_err)

# 		assert x_restored.shape == x.shape
# 		assert torch.equal(x_restored, x)



def test_hbehavemae_verbose_temporal_token_sweep_rebuild_model():
	"""Verbose mini integration test: rebuild model for each token count and run end-to-end."""
	torch.manual_seed(2)
	device = torch.device("cpu")

	_print_header("HBehaveMAE temporal sweep (rebuild model)")

	for temporal_len in [24, 48, 96]:
		print("\n--- Building model for temporal_len:", temporal_len)

		model = HBehaveMAE(
			input_size=(4, 96),
			in_chans=1,
			embed_dim=16,
			num_heads=2,
			out_embed_dims=(16, 32, 64),
			stages=(3, 1, 2),
			q_strides=[(1, 2, 2), (2, 1, 4)],
			mask_unit_attn=(True, False, False),
			patch_kernel=(2, 1, 2),
			patch_stride=(2, 1, 2),
			patch_padding=(0, 0, 0),
			decoder_embed_dim=32,
			decoder_depth=1,
			decoder_num_heads=2,
			decoding_strategy="single",
			norm_loss=False,
		).to(device)
		model.train()

		x = torch.randn(17, temporal_len, 4, 96, device=device)
		_print_tensor_info("input_x", x)

		with torch.no_grad():
			loss, pred, label, _, mask = model(x, mask_ratio=0.5)

		print("loss:", float(loss.item()))
		print("masked_ratio:", float((~mask).float().mean().item()))




def test_hbehavemae_same_model_two_temporal_lengths_verbose_xfail():
	"""Diagnostic xfail: same model, two temporal lengths (target for token-agnostic process)."""
	torch.manual_seed(3)
	device = torch.device("cpu")

	_print_header("HBehaveMAE same-model variable token count diagnostic")

	model = HBehaveMAE(
		input_size=(1, 24),
		in_chans=1,
		embed_dim=16,
		num_heads=2,
		out_embed_dims=(16, 32, 64, 98),
		stages=(1, 3, 1, 2),
		q_strides=[(2, 1, 2), (2, 1, 3), (2, 1, 2)],
		mask_unit_attn=(True, False, False, True),
		patch_kernel=(2, 1, 2),
		patch_stride=(2, 1, 2),
		patch_padding=(0, 0, 0),
		decoder_embed_dim=32,
		decoder_depth=3,
		decoder_num_heads=2,
		decoding_strategy="single",
		norm_loss=False,
	).to(device)
	model.train()

	x_48 = torch.randn(1, 128, 1, 24, device=device)
	with torch.no_grad():
		loss_48, *_ = model(x_48, mask_ratio=0.5)
	print("loss @48 tokens:", float(loss_48.item()))

	x_96 = torch.randn(1, 256, 1, 24, device=device)
	print("About to run same model on temporal_len=96 (expected to fail before token-agnostic refactor).")
	with torch.no_grad():
		_ = model(x_96, mask_ratio=0.5)
