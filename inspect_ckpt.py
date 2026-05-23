import torch
ckpt = torch.load('./logs/sac_mlp_20260318_010513/best_model.pt', map_location='cpu', weights_only=False)
print("Actor state dict keys:")
for k, v in ckpt['actor_state_dict'].items():
    print(f'  {k}: {v.shape}')
