# ACP 算法完整讲解（落实版）

> 更新时间：2026-07-07  
> 适用对象：把 ACP 讲清楚、讲透彻、能落地的。

## 1. 一句话定义

**ACP（Adaptive Compliance Policy）是一个融合视觉与力觉的扩散模仿学习策略，能够同时预测参考位姿、虚拟目标位姿和接触柔顺刚度，让机器人在接触方向自动变软、在运动方向保持刚性。**

论文与项目：
- 论文：[arXiv 2410.09309](https://arxiv.org/abs/2410.09309)
- 项目页：[adaptive-compliance.github.io](https://adaptive-compliance.github.io/)
- 官方实现：[yifan-hou/adaptive_compliance_policy](https://github.com/yifan-hou/adaptive_compliance_policy)

## 2. ACP 要解决什么问题

接触任务（翻转、擦拭、装配）存在刚柔矛盾：
- 刚度高：轨迹准，但容易因微小误差产生大接触力
- 刚度低：安全，但轨迹会漂

传统视觉策略一般只预测位置，不预测柔顺参数，导致要么硬碰硬、要么全程太软。ACP 的目标是**时空可变柔顺**：只在该软的方向软，该硬的方向仍硬。

## 3. 核心思想（三条）

1. **只在力方向降刚度**：把接触力方向看作柔顺主轴
2. **虚拟目标替代直接力控**：
   \`x_virt = x_ref + K^{-1} f\`
3. **刚度随力连续变化**：从 `k_max` 平滑降到 `k_min`，避免突变抖动

## 4. 三个关键公式

1) 力方向单位向量
\`u = f / ||f||\`

2) 方向化刚度
\`K = S · diag(k_low, k_high, k_high) · S^-1\`

3) 虚拟目标
\`x_virt = x_ref + K^{-1} · f\`

工程含义：
- `x_ref`：理想轨迹
- `x_virt`：接触时真正跟踪的目标
- `k_low`：柔顺方向刚度（由策略预测）

## 5. 网络输入输出（19 维输出要会讲）

输入（多模态）：
- RGB 图像历史帧
- 六维力时序（FFT/TCN 编码）
- 末端位姿历史帧

输出（单臂 19 维）：
- 参考位姿 9D
- 虚拟目标位姿 9D
- 刚度幅值 1D（`k_low`）

## 6. 为什么 ACP 有效

- 接触冲突主要发生在法向方向，ACP 只在该方向退让
- 其他方向维持高刚度，轨迹不漂
- 虚拟目标把“力控需求”转成“位置接口可执行”的命令

## 7. 论文结果

项目页和论文报告：ACP 在接触任务中相较刚性/固定柔顺基线有显著提升，整体提升幅度可超过 50%。

## 8. 当前仓库里的对应实现

- 标签/虚拟目标估计：`adaptive_compliance_policy/PyriteEnvSuites/scripts/postprocess_add_virtual_target_label.py`
- 估计器核心：`adaptive_compliance_policy/PyriteUtility/planning_control/compliance_helpers.py`
- 推理重建刚度：`adaptive_compliance_policy/PyriteEnvSuites/env_runners/virtual_target_real_env_runner.py`
- 你已完成 Demo：`demo/virtual_target_stiffness_demo.py`

## 9. 当前 demo 的验证结果（已跑通）

来自 `demo/output/demo_summary.txt`：
- Steps: 400
- Contact start: 2.20 s
- Peak force: 12.09 N
- Stiffness range: [200, 5000] N/m
- Peak 时刻平移刚度特征值：[200, 5000, 5000]

这正好验证了“单方向柔顺，其余方向高刚度”的 ACP 核心机制。

## 10. i7 Pro 落地怎么讲

你们没有 UR5e 不影响做核心复现：
- 可先用现有控制栈完成虚拟目标与变刚度验证
- 后续在 RTX 4090 上训练策略
- 再做 i7 桥接层，把 ACP 输出映射到你们执行接口

配套文档见：`docs/I7_ACP_ADAPTATION.zh.md`。

## 11. 高频问答

**Q1：ACP 相比 Diffusion Policy 新在哪里？**  
A：不仅预测位姿，还预测柔顺刚度；并融合力觉特征。

**Q2：为什么要虚拟目标？**  
A：把力控转成位置可执行命令，适配大多数工业控制接口。

**Q3：为什么不是全维统一软？**  
A：统一软会丢轨迹；ACP 只在冲突方向软，兼顾安全和精度。

## 12. 下一步（当前阶段）

- 对照 demo 精读 `VirtualTargetEstimator`
- 完整训练与结果归档见 `docs/REPRODUCTION.zh.md`、`docs/TRAINING_RESULTS_SPEC.zh.md`
- 真机迁移见 `docs/I7_ACP_ADAPTATION.zh.md`
