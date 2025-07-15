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
    
    说明：
        只对model里存在且在masks_dict中的卷积层权重应用掩码。
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if name in masks_dict:
                mask = masks_dict[name].to(module.weight.device)
                # 权重置零
                with torch.no_grad():
                    module.weight.data.mul_(mask)
                # print(f"[{name}] 权重置零完成")


def prune_by_scores(scores_dict, prune_ratio=0.5, unit_size=64, existing_masks_dict=None):
    """
    基于得分分组剪枝，支持已有掩码基础上继续剪到目标稀疏率。

    参数：
        scores_dict: dict[layer_name] = score_tensor，形状与权重相同
        prune_ratio: float，目标总稀疏率（包含已有为0的位置）
        unit_size: int，每组多少个元素（连续切片大小）
        existing_masks_dict: dict[layer_name] = mask_tensor（已有的剪枝掩码）

    返回：
        masks_dict: dict[layer_name] = 新的掩码Tensor，和权重形状相同
    """
    masks_dict = {}

    for layer_name, score_tensor in scores_dict.items():
        total_weights = score_tensor.numel()
        if total_weights < unit_size:
            # 太小的层直接全保留
            masks_dict[layer_name] = torch.ones_like(score_tensor)
            print(f"[{layer_name}] 权重数 {total_weights} < {unit_size}, 不剪枝")
            continue

        score_flat = score_tensor.view(-1)
        # 初始掩码全部为1
        if existing_masks_dict is not None and layer_name in existing_masks_dict:
            mask_flat = existing_masks_dict[layer_name].view(-1).clone()
        else:
            mask_flat = torch.ones_like(score_flat)

        # 已剪掉的权重数
        num_already_zero = (mask_flat == 0).sum().item()

        # 目标剪枝总数
        total_prune_num = int(total_weights * prune_ratio)
        remaining_prune_num = max(total_prune_num - num_already_zero, 0)

        if remaining_prune_num == 0:
            # 已达到目标剪枝率，无需再剪
            masks_dict[layer_name] = mask_flat.view(score_tensor.shape)
            continue

        # 分组数量（不够整除最后一组可能小于unit_size）
        num_groups = (total_weights + unit_size - 1) // unit_size

        # 均分剪枝数量
        base_per_group = remaining_prune_num // num_groups
        remainder = remaining_prune_num % num_groups

        for g in range(num_groups):
            start = g * unit_size
            end = min(start + unit_size, total_weights)

            group_scores = score_flat[start:end]
            group_mask = mask_flat[start:end]

            valid_indices = torch.nonzero(group_mask == 1, as_tuple=True)[0]
            num_valid = valid_indices.numel()

            prune_in_group = base_per_group + (1 if g < remainder else 0)
            prune_in_group = min(prune_in_group, num_valid)

            if prune_in_group > 0 and num_valid > 0:
                valid_scores = group_scores[valid_indices]
                to_prune = torch.topk(valid_scores, prune_in_group, largest=False).indices
                prune_indices = valid_indices[to_prune]

                # 通过索引直接修改原掩码
                indices_in_mask_flat = start + prune_indices
                mask_flat[indices_in_mask_flat] = 0

        masks_dict[layer_name] = mask_flat.view(score_tensor.shape)

    return masks_dict

def prune_single_layer_by_score(scores_dict, target_layer, prune_ratio=0.5, unit_size=64):
    """
    只对目标层做剪枝，其余层不剪。
    
    参数：
        scores_dict: 所有层的得分
        target_layer: 要剪枝的层名（string）
        prune_ratio: 剪枝比例
        unit_size: 分组单位
    
    返回：
        masks_dict: 所有层的掩码（只有 target_layer 被剪）
    """
    masks_dict = {}
    for layer_name, score_tensor in scores_dict.items():
        if layer_name != target_layer:
            # 不剪的层全保留
            masks_dict[layer_name] = torch.ones_like(score_tensor)
            continue

        total_weights = score_tensor.numel()
        if total_weights < unit_size:
            masks_dict[layer_name] = torch.ones_like(score_tensor)
            print(f"[{layer_name}] 权重数 {total_weights} < {unit_size}, 不剪枝")
            continue

        score_flat = score_tensor.view(-1)
        num_groups = total_weights // unit_size
        mask_flat = torch.ones_like(score_flat)

        for g in range(num_groups):
            start = g * unit_size
            end = start + unit_size
            group_scores = score_flat[start:end]

            prune_num = int(unit_size * prune_ratio)
            if prune_num == 0:
                continue

            prune_indices = torch.topk(group_scores, prune_num, largest=False).indices
            mask_flat[start:end][prune_indices] = 0

        mask = mask_flat.view(score_tensor.shape)
        masks_dict[layer_name] = mask

        kept = mask.sum().item()
        print(f"[{layer_name}] 只剪此层，共{total_weights}个参数，保留{kept}个")

    return masks_dict

def compute_current_sparsity_paper(epoch, start_epoch, end_epoch, final_sparsity, initial_sparsity=0.0):
    """
    论文逐步剪枝非线性三次幂调度公式
    s_t = s_f + (s_i - s_f) * (1 - (t - t0)/(nΔt))^3
    """
    if epoch < start_epoch:
        return initial_sparsity
    elif epoch > end_epoch:
        return final_sparsity
    else:
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)  # 0~1
        return final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
    
def update_masks_groupwise_64(model, scores_dict, epoch, start_epoch, end_epoch, final_sparsity, existing_masks_dict=None):
    """
    Wanda++ group-wise 64元素剪枝（非线性逐步稀疏）

    参数:
        model: PyTorch模型
        scorer: WandaPlusElementwiseScorer实例，包含累积分数
        epoch: 当前训练epoch
        start_epoch: 剪枝开始epoch
        end_epoch: 剪枝结束epoch
        final_sparsity: 目标稀疏率
        existing_masks_dict: 上轮掩码（用于累积剪枝）

    返回:
        masks_dict: 当前epoch掩码字典 {层名: mask_tensor}
    """
    # 没到间隔就直接复用旧 mask

    # 使用三次方公式平滑递增稀疏率
    # s_t = s_f + (s_i - s_f) * (1 - progress)^3
    current_sparsity = compute_current_sparsity_paper(
        epoch,
        start_epoch,
        end_epoch,
        final_sparsity,
    )
    print(current_sparsity,'================')
    masks_dict = {}

    unit_size = 64

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in scores_dict:
            score = scores_dict[name].to(module.weight.device)
            weight = module.weight.data

            flat_score = score.view(-1)
            num_units = flat_score.numel() // unit_size

            if num_units == 0:
                # 权重数量不足一组，跳过
                continue

            flat_score = flat_score[:num_units * unit_size]  # 截断
            grouped_score = flat_score.view(num_units, unit_size).mean(dim=1)

            # 保持已有为0的组不恢复
            if existing_masks_dict is not None and name in existing_masks_dict:
                prev_mask = existing_masks_dict[name].view(-1)[:num_units * unit_size]
                prev_mask = prev_mask.view(num_units, unit_size)
                valid_mask = (prev_mask.mean(dim=1) > 0).float()
                grouped_score *= valid_mask

            # top-k组保留
            k = max(1, int((1 - current_sparsity) * num_units))
            if k == 0:
                # 防止k=0导致错误，全部剪除
                group_mask = torch.zeros_like(grouped_score)
            else:
                topk_val, _ = torch.topk(grouped_score, k, sorted=False)
                threshold = topk_val[-1]
                group_mask = (grouped_score >= threshold).float()

            # 展开mask至每个元素
            group_mask = group_mask.view(-1, 1).expand(-1, unit_size)

            # 处理尾部剩余元素（如果存在）
            tail = score.numel() % unit_size
            if tail > 0:
                tail_mask = torch.ones(tail, device=score.device)
                full_mask = torch.cat([group_mask.reshape(-1), tail_mask], dim=0)
            else:
                full_mask = group_mask.reshape(-1)

            full_mask = full_mask.view_as(score)
            module.weight.data *= full_mask
            masks_dict[name] = full_mask

    return masks_dict

def apply_mask_to_grads_and_state(optimizer, masks_dict):
    """在 optimizer.step() 前屏蔽已剪枝权重的梯度和动量。

    新版实现 **不依赖 `param_name`**，而是按以下顺序寻找掩码：

    1. 先用参数对象本身作为 key（推荐：构造 masks_dict 时直接用 param）
    2. 再退回旧版的 name 键，保证向后兼容
    """

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue

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

            # ① 屏蔽梯度
            p.grad.mul_(mask)

            # ② 屏蔽动量 / EMA 等状态
            state = optimizer.state[p]
            for key in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
                if key in state:
                    state[key].mul_(mask)

def compute_sparsity(masks_dict: dict):
    total = 0
    zero = 0
    for mask in masks_dict.values():
        total += mask.numel()
        zero += (mask == 0).sum().item()
    return zero / total

def prune_by_weight_magnitude_groupwise_64(model: nn.Module, epoch, start_epoch, end_epoch, final_sparsity, unit_size: int = 64, existing_masks_dict: Optional[Dict[str, torch.Tensor]] = None):
    """按照权重绝对值做 group-wise 64 剪枝。

    参数:
        model:     PyTorch 模型（遍历其中 Conv2d 层）
        prune_ratio: 目标稀疏率 (0~1)，例如 0.5 表示剪掉 50% 的 group
        unit_size: 每个剪枝单元内包含的元素个数，默认 64
        existing_masks_dict:  前一轮掩码（继续在其基础上裁剪），可为 None

    返回:
        masks_dict: dict[layer_name] = mask_tensor (0/1)
    """
    prune_ratio = compute_current_sparsity_paper(epoch, start_epoch, end_epoch, final_sparsity)
    assert 0.0 <= prune_ratio <= 1.0, "prune_ratio 应在 0~1 之间"
    masks_dict: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        weight = module.weight.detach()
        abs_weight = weight.abs()
        total_elems = abs_weight.numel()

        # 不足一个 group 直接跳过
        if total_elems < unit_size:
            masks_dict[name] = torch.ones_like(weight, dtype=weight.dtype, device=weight.device)
            continue

        # 展平成 1-D，并取整批 group（尾部不足 unit_size 的元素暂时不参与排序，将在后面直接保留）
        flat_w = abs_weight.view(-1)
        num_full_groups = flat_w.numel() // unit_size
        head_flat = flat_w[: num_full_groups * unit_size]
        tail_flat = flat_w[num_full_groups * unit_size :]

        # 计算每个 group 的平均绝对值
        grouped = head_flat.view(num_full_groups, unit_size)
        group_mean = grouped.mean(dim=1)

        # 保证已有为 0 的 group 不被复活
        if existing_masks_dict is not None and name in existing_masks_dict:
            prev_mask = existing_masks_dict[name].view(-1)[: num_full_groups * unit_size]
            prev_mask = prev_mask.view(num_full_groups, unit_size)
            valid_mask = (prev_mask.mean(dim=1) > 0).float()  # 0 表示已剪掉
            group_mean = group_mean * valid_mask  # 已经被剪的 group 均值变 0，保证继续保持 0

        # 计算需保留的 group 数量
        k_keep = int((1.0 - prune_ratio) * num_full_groups)
        k_keep = max(k_keep, 1)

        if k_keep == 0:
            group_keep_mask = torch.zeros_like(group_mean)
        else:
            top_vals, _ = torch.topk(group_mean, k_keep, largest=True, sorted=False)
            threshold = top_vals.min()
            group_keep_mask = (group_mean >= threshold).float()

        # 再与 valid_mask 取交集，彻底防止已剪掉的 group 被“复活”
        if existing_masks_dict is not None and name in existing_masks_dict:
            group_keep_mask = group_keep_mask * valid_mask

        # 展开到元素级别
        expanded_mask = group_keep_mask.view(-1, 1).expand(-1, unit_size).reshape(-1)

        # 将尾部不足 unit_size 的元素全部保留
        if tail_flat.numel() > 0:
            tail_mask = torch.ones_like(tail_flat)
            full_mask_flat = torch.cat([expanded_mask, tail_mask], dim=0)
        else:
            full_mask_flat = expanded_mask

        full_mask = full_mask_flat.view_as(weight).to(weight.device)
        masks_dict[name] = full_mask

    return masks_dict



import os
def save_pruned_model(model, masks_dict, filename, ema=None):
    if ema:
        model = deepcopy(ema)
        print('====================copy to model')
    apply_mask_to_model_weights(model, masks_dict)
    save_path = os.path.join('/home/zhengxiuzhang/ultralytics-main/pruned_model', filename)
    torch.save(model.state_dict(), save_path)

def update_masks_and_clear_scorer(self, scorer, start_epoch, end_epoch, final_sparsity, previous_masks_dict):
    """先按权重幅度做一次 group-wise 64 剪枝，再按得分更新 mask，最后清空 scorer。

    这样便可在 trainer 中只保留一次函数调用，减少重复代码。
    """
    # 2. 根据 Wanda++ 得分进一步更新 mask（不"复活"已剪掉的权重）
    scores_dict = scorer.compute_final_scores()
    masks = update_masks_groupwise_64(
        model=self.model.module,
        scores_dict=scores_dict,
        epoch=self.epoch,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        final_sparsity=final_sparsity,
        existing_masks_dict=previous_masks_dict,
    )

    # 3. 清理 scorer 相关缓存，便于下一 epoch 重新累积
    scorer.scores_sum.clear()
    scorer.forward_acts_sum.clear()
    scorer.forward_acts_count.clear()

    return masks

import math
from typing import List, Tuple

def adjust_phase_lr(
        epoch: int,
        optimizer,
        total_epochs: int = 100,      # 训练总 epoch
        prune_epochs: int = 70,       # 剪枝期（含微调）长度
        plateau_lr: float = 1e-3,     # 初始 LR（剪枝初期）
        ft_peak_lr: float = 1e-4,     # 微调起始 LR（剪枝后期末）
        ft_min_lr: float = 1e-5,      # 微调最低 LR
):
    """
    新策略：
    剪枝阶段：线性下降学习率（plateau_lr -> ft_peak_lr）
    微调阶段：从 ft_peak_lr 开始余弦退火到 ft_min_lr
    """

    if epoch < prune_epochs:
        # 阶段1：剪枝阶段，线性下降 LR
        progress = epoch / max(1, prune_epochs)
        lr = plateau_lr + (ft_peak_lr - plateau_lr) * progress  # 注意：ft_peak_lr < plateau_lr
    else:
        # 阶段2：微调阶段，余弦退火
        progress = (epoch - prune_epochs) / max(1, total_epochs - prune_epochs)
        lr = ft_min_lr + 0.5 * (ft_peak_lr - ft_min_lr) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = lr