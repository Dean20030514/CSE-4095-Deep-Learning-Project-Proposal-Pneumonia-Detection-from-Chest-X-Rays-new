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

    def _forward_hook(self, _module, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        生成 Grad-CAM 热力图
        
        Args:
            input_tensor: 输入图像张量 (1, C, H, W) 或 (C, H, W)
            target_class: 目标类别索引
        
        Returns:
            cam: 归一化的 CAM 热力图 (H, W)
        
        Example:
            >>> gradcam = GradCAM(model, 'layer4')
            >>> cam = gradcam(input_image, target_class=1)
            >>> # cam shape: (H, W), values in [0, 1]
        """
        # 确保输入是 4D (batch_size=1)
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # 记录输入尺寸用于后续上采样
        input_size = input_tensor.shape[-2:]  # (H, W)
        
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)
        
        # 清空之前的梯度
        self.model.zero_grad(set_to_none=True)
        
        # 反向传播目标类别的得分
        target_score = output[0, target_class]
        target_score.backward()
        
        # 检查是否成功捕获梯度和激活
        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Failed to capture gradients or activations. "
                "Check if the target layer name is correct."
            )
        
        # 计算权重：对梯度进行全局平均池化
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # 加权求和生成 CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        
        # 只保留正值（ReLU）
        cam = F.relu(cam)
        
        # 上采样到输入图像尺寸
        cam = F.interpolate(
            cam, 
            size=input_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 归一化到 [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        # 返回 2D 热力图 (H, W)
        return cam.squeeze().cpu()
