import torch
import torch.nn as nn
from typing import Dict, Optional
from copy import copy, deepcopy
import math

class WandaPlusElementwiseScorer:
    def __init__(self, model, alpha=0.5, device='cuda'):
        """
        Wanda++ 元素级得分计算器（支持激活和梯度累积）

        参数:
            model: 需要计算得分的 PyTorch 模型
            alpha: Wanda++ 中的超参数
            device: 使用设备
        """
        self.model = model.to(device)
        self.alpha = alpha
        self.device = device

        self.forward_acts_sum = {}    # 累积输入激活，形状[C,H,W]
        self.forward_acts_count = {}  # 累积样本数

        self.scores_sum = {}          # 累积每个batch计算的得分
        self.hooks = []

    def _save_forward_hook(self, name):
        def hook(module, input, output):
            x = input[0].detach()  # shape: [B, C, H, W]

            # 对 batch 维度求和，累计激活
            x_sum = x.sum(dim=0)  # [C, H, W]

            if name not in self.forward_acts_sum:
                self.forward_acts_sum[name] = x_sum.clone()
                self.forward_acts_count[name] = x.shape[0]
            else:
                self.forward_acts_sum[name] += x_sum
                self.forward_acts_count[name] += x.shape[0]
        return hook

    def register_hooks(self):
        """
        注册所有卷积层的前向钩子，累积输入激活
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                h = module.register_forward_hook(self._save_forward_hook(name))
                self.hooks.append(h)

    def remove_hooks(self):
        """
        移除所有注册的钩子
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def accumulate_batch_scores(self):
        """
        对当前batch，计算并累积Wanda++得分（基于当前梯度和激活均值）
        需要保证前向激活和反向梯度已计算完成
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.weight.grad is None:
                    # 当前batch该层无梯度，跳过
                    continue
                if name not in self.forward_acts_sum or self.forward_acts_count[name] == 0:
                    print('act================')
                    # 激活缺失，跳过
                    continue

                weight = module.weight.detach()
                grad = module.weight.grad.detach()

                # 多个batch累积激活均值
                X_mean = self.forward_acts_sum[name] / self.forward_acts_count[name]  # [C,H,W]

                # 计算通道L2范数: shape [C]
                X_j_norm = torch.sqrt((X_mean ** 2).sum(dim=[1, 2]))

                # 调整形状用于广播: [1, C, 1, 1]
                X_j_norm_expand = X_j_norm.view(1, -1, 1, 1)

                # Wanda++元素级得分
                batch_score = (self.alpha * grad.abs() + X_j_norm_expand) * weight.abs()

                if name not in self.scores_sum:
                    self.scores_sum[name] = batch_score.clone()
                else:
                    self.scores_sum[name] += batch_score

    def compute_final_scores(self):
        """
        训练结束后，计算最终累积得分并返回
        返回:
            dict: {layer_name: 累积得分张量(cpu)}
        """
        return {k: v.cpu() for k, v in self.scores_sum.items()}
    
def apply_mask_to_model_weights(model, masks_dict):
    """
    用掩码把对应层的权重元素置零。
    
    参数：
        model: 你的YOLOv8模型
        masks_dict: dict，键是层名，值是掩码Tensor，形状和对应权重相同，元素是0/1
                   如果为None，则跳过掩码应用
    
    说明：
        只对model里存在且在masks_dict中的卷积层权重应用掩码。
    """
    if masks_dict is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if name in masks_dict:
                mask = masks_dict[name].to(module.weight.device)
                # 权重置零
                with torch.no_grad():
                    module.weight.data.mul_(mask)
                # print(f"[{name}] 权重置零完成")



def compute_current_sparsity_step(step, start_step, end_step, final_sparsity, initial_sparsity=0.0):
    if step < start_step:
        return initial_sparsity
    elif step > end_step:
        return final_sparsity
    else:
        progress = (step - start_step) / (end_step - start_step)
        return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
    


def reset_optimizer_state_for_pruned_weights(optimizer, masks_dict):
    """彻底重置被剪枝权重的优化器状态，确保它们不会影响训练。
    
    这个函数应该在剪枝后立即调用，确保被剪枝的权重完全停止更新。
    
    参数：
        optimizer: 优化器
        masks_dict: 掩码字典，如果为None则跳过处理
    """
    if masks_dict is None:
        return

    for group in optimizer.param_groups:
        for p in group["params"]:
            # 优先使用参数对象本身作为 key
            mask = masks_dict.get(p, None)

            # 兼容旧版：回退到 name 查找
            if mask is None:
                name = getattr(p, "param_name", None)
                if name is not None:
                    mask = masks_dict.get(name, None)

            if mask is None:
                continue  # 找不到对应 mask，跳过

            mask = mask.to(p.device)
            zero_mask = (mask == 0)  # 被剪枝的位置

            if not zero_mask.any():
                continue  # 没有剪枝位置，跳过

            # 重置优化器状态中被剪枝的位置
            state = optimizer.state[p]
            
            # 对于Adam优化器
            if "exp_avg" in state:
                state["exp_avg"][zero_mask] = 0.0
            if "exp_avg_sq" in state:
                state["exp_avg_sq"][zero_mask] = 0.0
            
            # 对于SGD优化器
            if "momentum_buffer" in state:
                state["momentum_buffer"][zero_mask] = 0.0
            
            # 对于其他可能的优化器状态
            for key in state:
                if isinstance(state[key], torch.Tensor) and state[key].shape == p.shape:
                    state[key][zero_mask] = 0.0


def compute_sparsity(masks_dict: dict):
    total = 0
    zero = 0
    for mask in masks_dict.values():
        total += mask.numel()
        zero += (mask == 0).sum().item()
    return zero / total


def update_masks_and_clear_scorer(self, scorer, step, total_steps, prune_steps, final_sparsity, previous_masks_dict, prune_fre=100):
    """先按权重幅度做一次 group-wise 64 剪枝，再按得分更新 mask，最后清空 scorer。

    这样便可在 trainer 中只保留一次函数调用，减少重复代码。
    """
    # 检查是否需要更新mask（每prune_fre步更新一次）
    if step % prune_fre != 0 or step >= total_steps:
        return previous_masks_dict
    
    # 计算当前稀疏率
    if step < prune_steps:
        current_sparsity = compute_current_sparsity_step(step, 0, prune_steps, final_sparsity, initial_sparsity=0.0)
    else:
        current_sparsity = final_sparsity
    
    print(f"[Step Prune] Step {step}: sparsity {current_sparsity:.4f}")
    
    # 2. 根据 Wanda++ 得分进一步更新 mask（不"复活"已剪掉的权重）
    if self.args.prune_method == 'wanda':
        scores_dict = scorer.compute_final_scores()
        masks = update_masks_asic_style(self.model.module, scores_dict, step, total_steps, prune_steps, current_sparsity, group_size_value=64, dtype='int8', existing_masks_dict=previous_masks_dict)
        scorer.scores_sum.clear()
        scorer.forward_acts_sum.clear()
        scorer.forward_acts_count.clear()
    elif self.args.prune_method == 'abs':
        masks = update_masks_asic_style_magnitude(self.model.module, step, total_steps, prune_steps, current_sparsity, group_size_value=64, dtype='int8', existing_masks_dict=previous_masks_dict)
    else:
        masks = None
    return masks

import math
from typing import List, Tuple


def adjust_lr_by_step(batch_size,step, optimizer, total_steps, base_lr=5e-4, min_lr=5e-6):
    prune_steps = int(total_steps * 0.8)
    base_lr=base_lr*batch_size/8
    min_lr=min_lr*batch_size/8
    if step < prune_steps:
        lr = base_lr
    else:
        decay_progress = (step - prune_steps) / (total_steps - prune_steps)
        lr = base_lr - (base_lr - min_lr) * decay_progress
    for pg in optimizer.param_groups:
        pg["lr"] = lr


import numpy as np

def np_topk(weight, k, axis=1):
    """
    perform topK based on np.argsort
    :param weight: to be sorted
    :param k: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(weight, axis=axis)
    
    # 处理不同维度的情况
    if weight.ndim == 1:
        # 1维数组，直接反转
        full_sort = full_sort[::-1]
    elif axis == 0:
        # 2维数组，axis=0时反转行
        full_sort = full_sort[::-1, :]
    else:
        # 2维数组，axis=1时反转列
        full_sort = full_sort[:, ::-1]
    
    return full_sort.take(np.arange(k), axis=axis)


def _block_sparsity_balance_inchannel(transpose_weight, keep_k):
    """
    ASIC风格的块稀疏平衡 - Input Channel版本
    
    参数:
        transpose_weight: 形状为 [H, W, out_ch, group_in_ch] 的权重
        keep_k: 保留的参数数量
    
    返回:
        mask: 形状与transpose_weight相同的掩码
    """
    # 重塑为 [H*W*out_ch, group_in_ch] 以便在input channel维度上操作
    original_shape = transpose_weight.shape
    reshape_weight = np.reshape(transpose_weight, [-1, transpose_weight.shape[-1]])
    
    
    # 每个位置(H*W*out_ch)分配相同的保留数量
    base_k = keep_k // reshape_weight.shape[0]  # 每个位置保留的input channel数
    remain_k = keep_k % reshape_weight.shape[0]  # 剩余的参数数量
    
    
    if remain_k > 0:
        # 如果有剩余，前remain_k个位置多保留1个
        index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k + 1))
    else:
        index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k))
    
    # 构建掩码
    dim1 = []
    dim2 = []
    for i, temp in enumerate(index.tolist()):
        for j in temp:
            dim1.append(i)
            dim2.append(j)
    
    mask = np.zeros(reshape_weight.shape)
    mask[dim1, dim2] = 1
    
    # 重塑回原始形状
    mask = mask.reshape(original_shape)
    mask = mask.astype(dtype=transpose_weight.dtype)
    
    
    return mask

def update_mask_asic_4d_torch(weight_tensor, keep_k, group_size_value=64, dtype='int8'):
    """
    ASIC风格的4D权重剪枝（PyTorch版本）- Input Channel分组
    
    参数:
        weight_tensor: PyTorch张量，形状为 [out_ch, in_ch, H, W]
        keep_k: 保留的参数数量
        group_size_value: 组大小，默认64
        dtype: 数据类型
    
    返回:
        mask: PyTorch张量，与weight_tensor形状相同的掩码
    """
    # 转换为numpy进行处理
    weight = weight_tensor.detach().cpu().numpy()
    
    if keep_k >= 1:
        # 从 [out_ch, in_ch, H, W] 转换为 [H, W, out_ch, in_ch] (将in_ch放到最后)
        h, w, i, o = weight.shape[2], weight.shape[3], weight.shape[1], weight.shape[0]
        transpose_weight = np.transpose(weight, [2, 3, 0, 1])  # [H, W, out_ch, in_ch]
        
        if transpose_weight.shape[0] == 1 and transpose_weight.shape[1] == 1:
            # 1x1卷积的情况
            transpose_weight = np.squeeze(transpose_weight, axis=(0, 1))  # [out_ch, in_ch]
            
            # 使用简化的2D处理 - 在input channel维度分组
            mask = update_mask_asic_2d_inchannel_torch(transpose_weight, keep_k, dtype, group_size_value=group_size_value)
            mask = np.reshape(mask, [h, w, o, i])
        else:
            # 常规卷积的情况 - 在input channel维度分组
            group_size = group_size_value
            if dtype in ['bf16', 'bfloat16']:
                group_size = group_size_value // 2
            
            temp1 = transpose_weight.shape[-1] // group_size  # in_ch维度的组数
            temp2 = transpose_weight.shape[-1] % group_size   # 剩余的in_ch
            
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            
            mask = np.ones([h, w, o, i])
            
            
            # 处理完整的组 - 在input channel维度
            if temp1 > 0:
                for idx in range(temp1):
                    start_ch = idx * group_size
                    end_ch = (idx + 1) * group_size
                    
                    # 提取当前组: [H, W, out_ch, group_in_ch]
                    transpose_weight_1 = transpose_weight[:, :, :, start_ch:end_ch]
                    mask_1 = _block_sparsity_balance_inchannel(transpose_weight_1, keep_k_1 // temp1)
                    mask[:, :, :, start_ch:end_ch] = mask_1
            
            # 处理剩余的通道
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size:]
                mask_2 = _block_sparsity_balance_inchannel(transpose_weight_2, keep_k_2)
                mask[:, :, :, temp1 * group_size:] = mask_2
        
        # 转换回原始形状 [out_ch, in_ch, H, W]
        mask = np.transpose(mask, [2, 3, 0, 1])
    else:
        mask = np.zeros_like(weight)
    
    # 转换回PyTorch张量
    return torch.from_numpy(mask).to(weight_tensor.device).type(weight_tensor.dtype)

def update_mask_asic_2d_torch(weight, keep_k, dtype, group_size_value=64):
    """
    ASIC风格的2D权重剪枝辅助函数
    """
    if keep_k >= 1:
        # 简化的2D处理逻辑
        reshape_weight = weight.reshape(-1)
        index = np_topk(np.abs(reshape_weight), keep_k, axis=0)
        mask = np.zeros(reshape_weight.shape)
        mask[index] = 1
        mask = mask.reshape(weight.shape)
        mask = mask.astype(weight.dtype)
    else:
        mask = np.zeros_like(weight)
    return mask

def update_mask_asic_2d_inchannel_torch(weight, keep_k, dtype, group_size_value=64):
    """
    ASIC风格的2D权重剪枝辅助函数 - Input Channel分组
    """
    if keep_k >= 1:
        # 简化的2D处理逻辑
        reshape_weight = weight.reshape(-1)
        index = np_topk(np.abs(reshape_weight), keep_k, axis=0)
        mask = np.zeros(reshape_weight.shape)
        mask[index] = 1
        mask = mask.reshape(weight.shape)
        mask = mask.astype(weight.dtype)
    else:
        mask = np.zeros_like(weight)
    return mask

def update_masks_asic_style(model, scores_dict, step, total_steps, prune_steps, current_sparsity, group_size_value=64, dtype='int8', existing_masks_dict=None):
    """
    ASIC风格的group-wise剪枝（模仿提供的代码逻辑，step级调度）

    参数:
        model: PyTorch模型
        scores_dict: Wanda++ 得分字典
        step: 当前step
        total_steps: 总step数
        prune_steps: 剪枝阶段step数（前80%）
        current_sparsity: 当前稀疏率
        group_size_value: 组大小，默认64
        dtype: 数据类型
        existing_masks_dict: 上轮掩码（用于累积剪枝）

    返回:
        masks_dict: 当前step掩码字典 {层名: mask_tensor}
    """
    # -------- 预热阶段：不进行剪枝 --------
    if step < 0:
        if existing_masks_dict is not None:
            return existing_masks_dict
        else:
            masks_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    masks_dict[name] = torch.ones_like(module.weight)
            return masks_dict

    # -------- 剪枝阶段/微调阶段 --------
    masks_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in scores_dict:
            weight = module.weight.data
            score = scores_dict[name].to(weight.device)
            # 保护关键层：完全跳过剪枝
            # critical_layers = ['model.0.conv', 'model.22.cv2.0.2', 'model.22.cv3.0.2', 'model.22.dfl.conv']
            critical_layers = ['model.22.dfl.conv']
            if name in critical_layers:
                masks_dict[name] = torch.ones_like(weight)
                continue
            total_params = weight.numel()
            keep_k = max(int(total_params * (1.0 - current_sparsity)), 1)
            mask = update_mask_asic_4d_torch(
                weight_tensor=score,
                keep_k=keep_k,
                group_size_value=group_size_value,
                dtype=dtype
            )
            if existing_masks_dict is not None and name in existing_masks_dict:
                prev_mask = existing_masks_dict[name]
                mask = mask * prev_mask
            module.weight.data *= mask
            masks_dict[name] = mask
    return masks_dict


def update_masks_asic_style_magnitude(model, step, total_steps, prune_steps, current_sparsity, group_size_value=64, dtype='int8', existing_masks_dict=None):
    """
    ASIC风格的group-wise剪枝（基于权重绝对值，step级调度）
    """
    if step < 0:
        if existing_masks_dict is not None:
            return existing_masks_dict
        else:
            masks_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    masks_dict[name] = torch.ones_like(module.weight)
            return masks_dict
    masks_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            # critical_layers = ['model.0.conv', 'model.22.cv2.0.2', 'model.22.cv3.0.2', 'model.22.dfl.conv']
            critical_layers = ['model.22.dfl.conv']
            if name in critical_layers:
                masks_dict[name] = torch.ones_like(weight)
                continue
            total_params = weight.numel()
            keep_k = max(int(total_params * (1.0 - current_sparsity)), 1)
            mask = update_mask_asic_4d_torch(
                weight_tensor=weight.abs(),
                keep_k=keep_k,
                group_size_value=group_size_value,
                dtype=dtype
            )
            if existing_masks_dict is not None and name in existing_masks_dict:
                prev_mask = existing_masks_dict[name]
                mask = mask * prev_mask
            module.weight.data *= mask
            masks_dict[name] = mask
    return masks_dict
