from typing import List
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        target_layer = dict([*self.model.named_modules()]).get(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found")
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def __call__(self, scores: torch.Tensor, idx: int) -> torch.Tensor:
        # scores: output logits (N, C)
        score = scores[:, idx].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over H,W
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(scores.shape[-1], scores.shape[-1]), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam.squeeze(0).squeeze(0)
