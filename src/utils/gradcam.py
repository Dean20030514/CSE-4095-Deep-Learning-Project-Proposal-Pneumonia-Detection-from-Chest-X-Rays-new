"""
Grad-CAM 可解释性工具

生成类激活映射(Class Activation Map)以解释模型预测。
"""
import torch
import torch.nn.functional as F
from typing import Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    生成热力图显示模型关注的图像区域。
    
    Attributes:
        model: PyTorch 模型
        target_layer_name: 目标层名称（通常是最后一个卷积层）
        
    Example:
        >>> gradcam = GradCAM(model, 'layer4')
        >>> cam = gradcam(input_image, target_class=1)
        >>> gradcam.remove_hooks()  # 使用完毕后清理
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str):
        """
        初始化 GradCAM。
        
        Args:
            model: PyTorch 模型
            target_layer_name: 目标层名称
            
        Raises:
            ValueError: 如果找不到指定的层
        """
        self.model = model.eval()
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._target_layer_name = target_layer_name
        
        # 找到目标层
        target_layer = dict([*self.model.named_modules()]).get(target_layer_name)
        if target_layer is None:
            raise ValueError(f"Layer {target_layer_name} not found in model. "
                           f"Available layers: {list(dict(self.model.named_modules()).keys())[:20]}...")
        
        # 注册 hooks 并保存 handles
        self._hooks.append(target_layer.register_forward_hook(self._forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, _module, _inp, out):
        """前向传播 hook，保存激活值"""
        self.activations = out.detach()

    def _backward_hook(self, _module, _grad_in, grad_out):
        """反向传播 hook，保存梯度"""
        self.gradients = grad_out[0].detach()
    
    def remove_hooks(self):
        """
        移除注册的 hooks。
        
        在不再需要 GradCAM 时调用以避免内存泄漏。
        """
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.gradients = None
        self.activations = None
    
    def __del__(self):
        """析构函数，自动清理 hooks"""
        self.remove_hooks()
    
    def __enter__(self):
        """支持 context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时清理 hooks"""
        self.remove_hooks()
        return False

    def __call__(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        生成 Grad-CAM 热力图
        
        Args:
            input_tensor: 输入图像张量 (1, C, H, W) 或 (C, H, W)
            target_class: 目标类别索引
        
        Returns:
            cam: 归一化的 CAM 热力图 (H, W)，值范围 [0, 1]
        
        Raises:
            RuntimeError: 如果无法捕获梯度或激活值
        
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
                f"Failed to capture gradients or activations for layer '{self._target_layer_name}'. "
                "This may happen if the layer doesn't participate in the forward pass "
                "or if gradient computation is disabled."
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
    
    def generate_batch(self, input_batch: torch.Tensor, target_classes: list) -> list:
        """
        为一批图像生成 Grad-CAM 热力图。
        
        Args:
            input_batch: 输入图像批次 (N, C, H, W)
            target_classes: 每个图像的目标类别列表
        
        Returns:
            热力图列表，每个元素形状为 (H, W)
        """
        cams = []
        for i in range(input_batch.size(0)):
            cam = self(input_batch[i:i+1], target_classes[i])
            cams.append(cam)
        return cams
