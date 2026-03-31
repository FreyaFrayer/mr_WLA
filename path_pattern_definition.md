# Path Pattern Definition

本文将路径分为三类局部几何模式：

- **Direction-consistent trend pattern**（同方向趋势型）
- **Multi-directional switching pattern**（多方向跳变型）
- **Directionally unstructured pattern / Random pattern**（随机型）

---

## 1. 统一数学表示

设当前点与未来窗口点为：

$$
\mathbf{p}_0,\mathbf{p}_1,\ldots,\mathbf{p}_H,\qquad \mathbf{p}_i\in\mathbb{R}^3
$$

定义位移与单位方向：

$$
\mathbf{d}_i=\mathbf{p}_i-\mathbf{p}_{i-1},\qquad i=1,\dots,H
$$

$$
\mathbf{u}_i=\frac{\mathbf{d}_i}{\lVert\mathbf{d}_i\rVert+\varepsilon}
$$

其中 $\varepsilon>0$ 为极小量。

---

## 2. 区分指标

### 2.1 方向一致性

$$
C=\frac{\left\lVert\sum_{i=1}^{H}\mathbf{u}_i\right\rVert}{H},\qquad C\in[0,1]
$$

- $C\approx 1$：方向高度一致
- $C\approx 0$：方向相互抵消

### 2.2 局部转向强度

$$
\theta_i=\arccos\!\left(\operatorname{clip}(\mathbf{u}_i^\top \mathbf{u}_{i+1},-1,1)\right),\qquad i=1,\dots,H-1
$$

$$
\bar{\theta}=\frac{1}{H-1}\sum_{i=1}^{H-1}\theta_i,\qquad
\theta_{\max}=\max_i \theta_i
$$

---

## 3. 三类 Pattern 定义

### A. 同方向趋势型

判据：

$$
C \ge \tau_C
\quad \text{and} \quad
\bar{\theta}\le \tau_\theta
$$

典型阈值：

$$
\tau_C = 0.85,\qquad \tau_\theta = 20^\circ \sim 30^\circ
$$

可选强化约束：

$$
\mathbf{u}_i^\top \mathbf{u}_{i+1} \ge \tau_{\cos},\qquad \forall i,
\quad \tau_{\cos}\in[0.7,0.9]
$$

### B. 多方向跳变型

满足以下任一条件：

$$
\theta_{\max}\ge \tau_{\text{jump}}
$$

或

$$
\exists i,\ \mathbf{u}_i^\top \mathbf{u}_{i+1}\le \tau_{\text{neg}}
$$

典型阈值：

- $\tau_{\text{jump}} = 60^\circ\sim90^\circ$
- $\tau_{\text{neg}} = 0$（更严格可取 $-0.3$）

通常伴随：

$$
C < \tau_C',\qquad \tau_C'\approx 0.7
$$

### C. 随机型

剩余类定义：

$$
\text{Random} \iff \text{not Trend and not Switching}
$$

可选“硬阈值”版本：

$$
\tau_C' \le C < \tau_C
\quad \text{and} \quad
\tau_\theta < \bar{\theta} < \tau_{\text{jump}}
$$

---

## 4. 三类对照

| 类型 | 方向一致性 $C$ | 转向角 $(\bar{\theta},\theta_{\max})$ | 几何特征 |
|---|---:|---:|---|
| 同方向趋势型 | 高 | 小 | 持续朝主方向推进 |
| 多方向跳变型 | 低或中低 | 大，且常有突变 | 明显拐弯、折返、切换 |
| 随机型 | 中等或低 | 中等 | 无稳定趋势、无显著结构性跳变 |

---

## 5. 未来 3 点窗口（简化版）

定义：

$$
\mathbf{d}_1=\mathbf{p}_1-\mathbf{p}_0,\quad
\mathbf{d}_2=\mathbf{p}_2-\mathbf{p}_1,\quad
\mathbf{d}_3=\mathbf{p}_3-\mathbf{p}_2
$$

$$
\theta_1=\angle(\mathbf{d}_1,\mathbf{d}_2),\qquad
\theta_2=\angle(\mathbf{d}_2,\mathbf{d}_3)
$$

$$
\bar{\theta}=\frac{\theta_1+\theta_2}{2},\qquad
\theta_{\max}=\max(\theta_1,\theta_2)
$$

实用三分类规则：

- 同方向趋势型：$\theta_1<30^\circ,\ \theta_2<30^\circ$
- 多方向跳变型：$\theta_{\max}>70^\circ$
- 随机型：其余情况

---

## 6. 可选增强：主方向投影单调性

定义主方向：

$$
\mathbf{v}=\frac{\sum_{i=1}^H \mathbf{d}_i}{\left\lVert\sum_{i=1}^H \mathbf{d}_i\right\rVert+\varepsilon}
$$

加入约束：

$$
\mathbf{d}_i^\top \mathbf{v} > 0,\qquad \forall i
$$

则同方向趋势型可写为：

$$
C \ge \tau_C,\qquad
\bar{\theta}\le \tau_\theta,\qquad
\mathbf{d}_i^\top \mathbf{v}>0\ \forall i
$$

---

## 7. 论文英文模板（精简）

> Let $\mathbf{p}_0,\mathbf{p}_1,\ldots,\mathbf{p}_H$ denote the current and future waypoints in a look-ahead window. Define increments $\mathbf{d}_i=\mathbf{p}_i-\mathbf{p}_{i-1}$ and normalized directions $\mathbf{u}_i=\mathbf{d}_i/(\lVert\mathbf{d}_i\rVert+\varepsilon)$. Directional coherence is measured by $C=\lVert\sum_{i=1}^{H}\mathbf{u}_i\rVert/H$, and local directional variation by $\theta_i=\arccos(\mathbf{u}_i^\top\mathbf{u}_{i+1})$. A segment is labeled as direction-consistent trend if coherence is high and turning is small; as multi-directional switching if large angular deviations occur; otherwise as directionally unstructured.
