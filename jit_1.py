import torch
from itertools import combinations
import time
import sys

# -------------------- 环境与数据 --------------------
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
HAS_COMPILE = TORCH_VERSION >= (2, 0)
print(f"PyTorch 版本: {torch.__version__}")
print(f"支持 torch.compile: {HAS_COMPILE}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 规模
M = 100
N = 1500
K = 5

torch.manual_seed(42)

# a_i: 实数 (M,)
a = torch.randn(M, dtype=torch.float32, device=device)

# b: 复数矩阵的实/虚拆分（只保留实数张量）
b_r = torch.randn(M, N, dtype=torch.float32, device=device)
b_i = torch.randn(M, N, dtype=torch.float32, device=device) + 0.1

# x_{j,k}: 实数输入 (N,K)
x = torch.randn(N, K, dtype=torch.float32, device=device)

# 复系数的实/虚拆分（仅用实数存储）
def make_c(n):
    r = torch.randn(n, device=device, dtype=torch.float32)
    i = torch.randn(n, device=device, dtype=torch.float32)
    return r, i

c0_r, c0_i = make_c(1)     # 标量存成1元素向量，后续用 .item()
c1_r, c1_i = make_c(K)

idx2 = list(combinations(range(K), 2))
idx3 = list(combinations(range(K), 3))
idx4 = list(combinations(range(K), 4))
idx5 = list(combinations(range(K), 5))

c2_r, c2_i = make_c(len(idx2))
c3_r, c3_i = make_c(len(idx3))
c4_r, c4_i = make_c(len(idx4))
c5_r, c5_i = make_c(len(idx5))

# 预先缓存索引张量（避免热路径创建）
idx2_tensor = torch.tensor(idx2, device=device, dtype=torch.long) if idx2 else torch.empty(0, 2, dtype=torch.long, device=device)
idx3_tensor = torch.tensor(idx3, device=device, dtype=torch.long) if idx3 else torch.empty(0, 3, dtype=torch.long, device=device)
idx4_tensor = torch.tensor(idx4, device=device, dtype=torch.long) if idx4 else torch.empty(0, 4, dtype=torch.long, device=device)
idx5_tensor = torch.tensor(idx5, device=device, dtype=torch.long) if idx5 else torch.empty(0, 5, dtype=torch.long, device=device)

# -------------------- 实/虚分离的前向与梯度 --------------------
def compute_X_parts(x_in,
                    c0_r, c0_i,
                    c1_r, c1_i,
                    c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                    idx2_t, idx3_t, idx4_t, idx5_t):
    """返回 (X_r, X_i)，均为 (N,) float32"""
    N = x_in.shape[0]
    X_r = torch.full((N,), c0_r.item(), dtype=torch.float32, device=x_in.device)
    X_i = torch.full((N,), c0_i.item(), dtype=torch.float32, device=x_in.device)

    # 一次项
    X_r += x_in @ c1_r
    X_i += x_in @ c1_i

    # 高次项：先算实数单项式 prod，再分别乘以实/虚系数
    if idx2_t.numel() > 0:
        prod2 = torch.prod(x_in[:, idx2_t], dim=2)  # (N,C2)
        X_r += prod2 @ c2_r
        X_i += prod2 @ c2_i
    if idx3_t.numel() > 0:
        prod3 = torch.prod(x_in[:, idx3_t], dim=2)  # (N,C3)
        X_r += prod3 @ c3_r
        X_i += prod3 @ c3_i
    if idx4_t.numel() > 0:
        prod4 = torch.prod(x_in[:, idx4_t], dim=2)  # (N,C4)
        X_r += prod4 @ c4_r
        X_i += prod4 @ c4_i
    if idx5_t.numel() > 0:
        prod5 = torch.prod(x_in[:, idx5_t], dim=2)  # (N,C5)
        X_r += prod5 @ c5_r
        X_i += prod5 @ c5_i

    return X_r, X_i

def compute_dX_parts_dx_original(x_in,
                                 c1_r, c1_i,
                                 c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                 idx2, idx3, idx4, idx5):
    """loop 版本 dX/dx（用于对比），返回 (dXr_dx, dXi_dx)，形状 (N,K)"""
    N, K = x_in.shape
    dXr_dx = torch.zeros(N, K, dtype=torch.float32, device=x_in.device)
    dXi_dx = torch.zeros_like(dXr_dx)

    # 一次项
    for k in range(K):
        dXr_dx[:, k] += c1_r[k]
        dXi_dx[:, k] += c1_i[k]

    # 二次项
    for i, (k1, k2) in enumerate(idx2):
        dXr_dx[:, k1] += c2_r[i] * x_in[:, k2]
        dXr_dx[:, k2] += c2_r[i] * x_in[:, k1]
        dXi_dx[:, k1] += c2_i[i] * x_in[:, k2]
        dXi_dx[:, k2] += c2_i[i] * x_in[:, k1]

    # 三次项
    for i, (k1, k2, k3) in enumerate(idx3):
        r, ii = c3_r[i], c3_i[i]
        x1, x2, x3 = x_in[:, k1], x_in[:, k2], x_in[:, k3]
        dXr_dx[:, k1] += r * x2 * x3
        dXr_dx[:, k2] += r * x1 * x3
        dXr_dx[:, k3] += r * x1 * x2
        dXi_dx[:, k1] += ii * x2 * x3
        dXi_dx[:, k2] += ii * x1 * x3
        dXi_dx[:, k3] += ii * x1 * x2

    # 四次项
    for i, (k1, k2, k3, k4) in enumerate(idx4):
        r, ii = c4_r[i], c4_i[i]
        x1, x2, x3, x4 = x_in[:, k1], x_in[:, k2], x_in[:, k3], x_in[:, k4]
        dXr_dx[:, k1] += r * (x2 * x3 * x4)
        dXr_dx[:, k2] += r * (x1 * x3 * x4)
        dXr_dx[:, k3] += r * (x1 * x2 * x4)
        dXr_dx[:, k4] += r * (x1 * x2 * x3)
        dXi_dx[:, k1] += ii * (x2 * x3 * x4)
        dXi_dx[:, k2] += ii * (x1 * x3 * x4)
        dXi_dx[:, k3] += ii * (x1 * x2 * x4)
        dXi_dx[:, k4] += ii * (x1 * x2 * x3)

    # 五次项
    for i, (k1, k2, k3, k4, k5) in enumerate(idx5):
        r, ii = c5_r[i], c5_i[i]
        x1, x2, x3, x4, x5 = x_in[:, k1], x_in[:, k2], x_in[:, k3], x_in[:, k4], x_in[:, k5]
        dXr_dx[:, k1] += r * (x2 * x3 * x4 * x5)
        dXr_dx[:, k2] += r * (x1 * x3 * x4 * x5)
        dXr_dx[:, k3] += r * (x1 * x2 * x4 * x5)
        dXr_dx[:, k4] += r * (x1 * x2 * x3 * x5)
        dXr_dx[:, k5] += r * (x1 * x2 * x3 * x4)
        dXi_dx[:, k1] += ii * (x2 * x3 * x4 * x5)
        dXi_dx[:, k2] += ii * (x1 * x3 * x4 * x5)
        dXi_dx[:, k3] += ii * (x1 * x2 * x4 * x5)
        dXi_dx[:, k4] += ii * (x1 * x2 * x3 * x5)
        dXi_dx[:, k5] += ii * (x1 * x2 * x3 * x4)

    return dXr_dx, dXi_dx

def compute_dX_parts_dx_vectorized(x_in,
                                   c1_r, c1_i,
                                   c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                   idx2_t, idx3_t, idx4_t, idx5_t):
    """向量化 dX/dx，返回 (dXr_dx, dXi_dx)"""
    N, K = x_in.shape
    dXr_dx = torch.zeros(N, K, dtype=torch.float32, device=x_in.device)
    dXi_dx = torch.zeros_like(dXr_dx)

    # 一次项
    dXr_dx += c1_r.unsqueeze(0)  # (N,K)
    dXi_dx += c1_i.unsqueeze(0)

    def add_order(indices_t, coeff_r, coeff_i, order):
        nonlocal dXr_dx, dXi_dx
        if indices_t.numel() == 0:
            return
        x_vals = x_in[:, indices_t]                # (N, C, order)
        full_prod = torch.prod(x_vals, dim=2)      # (N, C)
        for pos in range(order):
            x_pos = x_vals[:, :, pos]              # (N, C)
            # 避免除零：零位置的导数直接按“去掉该变量后的乘积”=0处理
            partial = torch.where(torch.abs(x_pos) > 1e-12,
                                  full_prod / x_pos,
                                  torch.zeros_like(full_prod))
            # 权重
            wr = coeff_r.unsqueeze(0) * partial    # (N, C)
            wi = coeff_i.unsqueeze(0) * partial
            var_idx = indices_t[:, pos].unsqueeze(0).expand(N, -1)  # (N, C)
            dXr_dx.scatter_add_(1, var_idx, wr)
            dXi_dx.scatter_add_(1, var_idx, wi)

    add_order(idx2_t, c2_r, c2_i, 2)
    add_order(idx3_t, c3_r, c3_i, 3)
    add_order(idx4_t, c4_r, c4_i, 4)
    add_order(idx5_t, c5_r, c5_i, 5)

    return dXr_dx, dXi_dx

def forward_pass_real(x_input):
    """纯实数前向（用于 autograd 基准）"""
    X_r, X_i = compute_X_parts(
        x_input, c0_r, c0_i, c1_r, c1_i,
        c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
        idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor
    )
    S_r = b_r @ X_r - b_i @ X_i
    S_i = b_r @ X_i + b_i @ X_r
    y = torch.sum(a * (S_r * S_r + S_i * S_i))
    return y

def compute_gradient_manual_real(x_in):
    """手动梯度（纯实数链式）"""
    X_r, X_i = compute_X_parts(
        x_in, c0_r, c0_i, c1_r, c1_i,
        c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
        idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor
    )
    S_r = b_r @ X_r - b_i @ X_i
    S_i = b_r @ X_i + b_i @ X_r

    w_r = 2.0 * a * S_r   # (M,)
    w_i = 2.0 * a * S_i   # (M,)

    dy_dX_r = w_r @ b_r + w_i @ b_i     # (N,)
    dy_dX_i = -w_r @ b_i + w_i @ b_r    # (N,)

    dXr_dx, dXi_dx = compute_dX_parts_dx_vectorized(
        x_in, c1_r, c1_i, c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
        idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor
    )
    grad = dy_dX_r.unsqueeze(1) * dXr_dx + dy_dX_i.unsqueeze(1) * dXi_dx
    return grad

# -------------------- 编译友好版本（签名固定 + 仅 float32） --------------------
def compute_X_parts_compiled(x_in: torch.Tensor,
                             c0_r: torch.Tensor, c0_i: torch.Tensor,
                             c1_r: torch.Tensor, c1_i: torch.Tensor,
                             c2_r: torch.Tensor, c2_i: torch.Tensor,
                             c3_r: torch.Tensor, c3_i: torch.Tensor,
                             c4_r: torch.Tensor, c4_i: torch.Tensor,
                             c5_r: torch.Tensor, c5_i: torch.Tensor,
                             idx2_t: torch.Tensor, idx3_t: torch.Tensor,
                             idx4_t: torch.Tensor, idx5_t: torch.Tensor):
    N = x_in.shape[0]
    X_r = torch.full((N,), float(c0_r.item()), dtype=torch.float32, device=x_in.device)
    X_i = torch.full((N,), float(c0_i.item()), dtype=torch.float32, device=x_in.device)

    X_r = X_r + (x_in @ c1_r)
    X_i = X_i + (x_in @ c1_i)

    if idx2_t.numel() > 0:
        prod2 = torch.prod(x_in[:, idx2_t], dim=2)
        X_r = X_r + prod2 @ c2_r
        X_i = X_i + prod2 @ c2_i
    if idx3_t.numel() > 0:
        prod3 = torch.prod(x_in[:, idx3_t], dim=2)
        X_r = X_r + prod3 @ c3_r
        X_i = X_i + prod3 @ c3_i
    if idx4_t.numel() > 0:
        prod4 = torch.prod(x_in[:, idx4_t], dim=2)
        X_r = X_r + prod4 @ c4_r
        X_i = X_i + prod4 @ c4_i
    if idx5_t.numel() > 0:
        prod5 = torch.prod(x_in[:, idx5_t], dim=2)
        X_r = X_r + prod5 @ c5_r
        X_i = X_i + prod5 @ c5_i

    return X_r, X_i

def compute_dX_parts_dx_compiled(x_in: torch.Tensor,
                                 c1_r: torch.Tensor, c1_i: torch.Tensor,
                                 c2_r: torch.Tensor, c2_i: torch.Tensor,
                                 c3_r: torch.Tensor, c3_i: torch.Tensor,
                                 c4_r: torch.Tensor, c4_i: torch.Tensor,
                                 c5_r: torch.Tensor, c5_i: torch.Tensor,
                                 idx2_t: torch.Tensor, idx3_t: torch.Tensor,
                                 idx4_t: torch.Tensor, idx5_t: torch.Tensor):
    N, K = x_in.shape
    dXr_dx = torch.zeros(N, K, dtype=torch.float32, device=x_in.device)
    dXi_dx = torch.zeros_like(dXr_dx)

    dXr_dx = dXr_dx + c1_r.unsqueeze(0)
    dXi_dx = dXi_dx + c1_i.unsqueeze(0)

    def add_order(indices_t, coeff_r, coeff_i, order):
        nonlocal dXr_dx, dXi_dx
        if indices_t.numel() == 0:
            return
        x_vals = x_in[:, indices_t]           # (N,C,order)
        full_prod = torch.prod(x_vals, dim=2) # (N,C)
        for pos in range(order):
            x_pos = x_vals[:, :, pos]
            partial = torch.where(torch.abs(x_pos) > 1e-12,
                                  full_prod / x_pos,
                                  torch.zeros_like(full_prod))
            wr = coeff_r.unsqueeze(0) * partial
            wi = coeff_i.unsqueeze(0) * partial
            var_idx = indices_t[:, pos].unsqueeze(0).expand(N, -1)
            dXr_dx.scatter_add_(1, var_idx, wr)
            dXi_dx.scatter_add_(1, var_idx, wi)

    add_order(idx2_t, c2_r, c2_i, 2)
    add_order(idx3_t, c3_r, c3_i, 3)
    add_order(idx4_t, c4_r, c4_i, 4)
    add_order(idx5_t, c5_r, c5_i, 5)

    return dXr_dx, dXi_dx

def compute_gradient_manual_compiled_real(x_in: torch.Tensor, a: torch.Tensor,
                                          b_r: torch.Tensor, b_i: torch.Tensor,
                                          c0_r: torch.Tensor, c0_i: torch.Tensor,
                                          c1_r: torch.Tensor, c1_i: torch.Tensor,
                                          c2_r: torch.Tensor, c2_i: torch.Tensor,
                                          c3_r: torch.Tensor, c3_i: torch.Tensor,
                                          c4_r: torch.Tensor, c4_i: torch.Tensor,
                                          c5_r: torch.Tensor, c5_i: torch.Tensor,
                                          idx2_t: torch.Tensor, idx3_t: torch.Tensor,
                                          idx4_t: torch.Tensor, idx5_t: torch.Tensor):
    X_r, X_i = compute_X_parts_compiled(
        x_in, c0_r, c0_i, c1_r, c1_i, c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
        idx2_t, idx3_t, idx4_t, idx5_t
    )

    S_r = b_r @ X_r - b_i @ X_i
    S_i = b_r @ X_i + b_i @ X_r

    w_r = 2.0 * a * S_r
    w_i = 2.0 * a * S_i

    dy_dX_r = w_r @ b_r + w_i @ b_i
    dy_dX_i = -w_r @ b_i + w_i @ b_r

    dXr_dx, dXi_dx = compute_dX_parts_dx_compiled(
        x_in, c1_r, c1_i, c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
        idx2_t, idx3_t, idx4_t, idx5_t
    )

    grad = dy_dX_r.unsqueeze(1) * dXr_dx + dy_dX_i.unsqueeze(1) * dXi_dx
    return grad

# -------------------- 创建编译版本 --------------------
HAS_JIT = False
'''
try:
    compute_gradient_jit = torch.jit.script(compute_gradient_manual_compiled_real)
    print("✓ TorchScript 编译成功")
    HAS_JIT = True
except Exception as e:
    print(f"✗ TorchScript 编译失败: {e}")
    HAS_JIT = False
'''
if HAS_COMPILE:
    try:
        compute_gradient_compile_default = torch.compile(compute_gradient_manual_compiled_real)
        compute_gradient_compile_reduce_overhead = torch.compile(compute_gradient_manual_compiled_real, mode="reduce-overhead")
        compute_gradient_compile_max_autotune = torch.compile(compute_gradient_manual_compiled_real, mode="max-autotune")
        print("✓ torch.compile 编译成功")
        HAS_TORCH_COMPILE = True
    except Exception as e:
        print(f"✗ torch.compile 编译失败: {e}")
        HAS_TORCH_COMPILE = False
else:
    HAS_TORCH_COMPILE = False

# -------------------- 性能测试工具 --------------------
def time_gradient_computation(method='manual', num_runs=10):
    times = []
    # 预热
    if method in ['jit', 'compile', 'compile_reduce', 'compile_max']:
        try:
            if method == 'jit' and HAS_JIT:
                _ = compute_gradient_jit(x, a, b_r, b_i,
                                         c0_r, c0_i, c1_r, c1_i,
                                         c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                         idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile' and HAS_TORCH_COMPILE:
                _ = compute_gradient_compile_default(x, a, b_r, b_i,
                                                     c0_r, c0_i, c1_r, c1_i,
                                                     c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                     idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile_reduce' and HAS_TORCH_COMPILE:
                _ = compute_gradient_compile_reduce_overhead(x, a, b_r, b_i,
                                                             c0_r, c0_i, c1_r, c1_i,
                                                             c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                             idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile_max' and HAS_TORCH_COMPILE:
                _ = compute_gradient_compile_max_autotune(x, a, b_r, b_i,
                                                          c0_r, c0_i, c1_r, c1_i,
                                                          c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                          idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
        except Exception as e:
            print(f"预热失败: {e}")
            return [], torch.zeros_like(x)

    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        try:
            if method == 'manual':
                gradient = compute_gradient_manual_real(x)
            elif method == 'jit' and HAS_JIT:
                gradient = compute_gradient_jit(x, a, b_r, b_i,
                                                c0_r, c0_i, c1_r, c1_i,
                                                c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile' and HAS_TORCH_COMPILE:
                gradient = compute_gradient_compile_default(x, a, b_r, b_i,
                                                            c0_r, c0_i, c1_r, c1_i,
                                                            c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                            idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile_reduce' and HAS_TORCH_COMPILE:
                gradient = compute_gradient_compile_reduce_overhead(x, a, b_r, b_i,
                                                                    c0_r, c0_i, c1_r, c1_i,
                                                                    c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                                    idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            elif method == 'compile_max' and HAS_TORCH_COMPILE:
                gradient = compute_gradient_compile_max_autotune(x, a, b_r, b_i,
                                                                 c0_r, c0_i, c1_r, c1_i,
                                                                 c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                                                                 idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor)
            else:  # autograd 基准
                x_temp = x.clone().requires_grad_(True)
                y = forward_pass_real(x_temp)
                y.backward()
                gradient = x_temp.grad
        except Exception as e:
            print(f"计算失败({method}): {e}")
            return [], torch.zeros_like(x)

        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        else:
            elapsed = (time.time() - start_time) * 1000.0

        times.append(elapsed)

    return times, gradient

# -------------------- 子模块性能对比（dX/dx） --------------------
def test_dX_dx_performance(use_vectorized=True, num_runs=10):
    times = []
    for _ in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()

        if use_vectorized:
            result_r, result_i = compute_dX_parts_dx_vectorized(
                x, c1_r, c1_i, c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                idx2_tensor, idx3_tensor, idx4_tensor, idx5_tensor
            )
        else:
            result_r, result_i = compute_dX_parts_dx_original(
                x, c1_r, c1_i, c2_r, c2_i, c3_r, c3_i, c4_r, c4_i, c5_r, c5_i,
                idx2, idx3, idx4, idx5
            )

        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        else:
            elapsed = (time.time() - start_time) * 1000.0
        times.append(elapsed)

    return times, (result_r, result_i)

# -------------------- 运行性能测试 --------------------
num_runs = 15
print(f"\n进行 {num_runs} 次测试以获得稳定的时间测量...")

print("\n" + "="*60)
print("dX/dx：loop vs 向量化（纯实数）")
print("="*60)
orig_times, (orig_r, orig_i) = test_dX_dx_performance(False, num_runs)
vec_times, (vec_r, vec_i) = test_dX_dx_performance(True, num_runs)

orig_avg = sum(orig_times) / len(orig_times)
vec_avg  = sum(vec_times)  / len(vec_times)

max_diff_dx = max(
    torch.max(torch.abs(orig_r - vec_r)).item(),
    torch.max(torch.abs(orig_i - vec_i)).item()
)
print(f"最大差异: {max_diff_dx:.2e}")
print("✓ 向量化 dX/dx 正确" if max_diff_dx < 1e-6 else "⚠️ 存在偏差")
print(f"原始平均: {orig_avg:.2f} ms | 向量化平均: {vec_avg:.2f} ms | 加速比: {orig_avg/vec_avg:.2f}x")

print("\n" + "="*60)
print("完整梯度：manual / autograd / JIT / compile（纯实数）")
print("="*60)

test_methods = ['manual', 'autograd', 'compile']
#if HAS_JIT: test_methods.append('jit')
#if HAS_TORCH_COMPILE: test_methods += ['compile', 'compile_reduce', 'compile_max']

results = {}
names = {
    'manual': '完全向量化手动梯度',
    'autograd': '自动微分梯度',
    'jit': 'TorchScript JIT',
    'compile': 'torch.compile (default)',
    'compile_reduce': 'torch.compile (reduce-overhead)',
    'compile_max': 'torch.compile (max-autotune)'
}
print("测试方法:", ', '.join(names[m] for m in test_methods), "\n")

for method in test_methods:
    print(f"测试 {names[method]}...")
    times, grad = time_gradient_computation(method, num_runs)
    times = times[1:]
    if times:
        avg = sum(times) / len(times)
        std = (sum((t-avg)**2 for t in times)/len(times))**0.5
        results[method] = {
            'times': times, 'avg_time': avg, 'std_time': std,
            'gradient': grad, 'min_time': min(times), 'max_time': max(times)
        }
        print(f"  平均: {avg:.2f} ± {std:.2f} ms | 范围: {min(times):.2f}-{max(times):.2f} ms | 范数: {torch.norm(grad).item():.6f}\n")
    else:
        print("  测试失败\n")

print("="*60)
print("性能对比与正确性（以 autograd 为基准）")
print("="*60)
if 'autograd' in results:
    base_grad = results['autograd']['gradient']
    base_time = results['autograd']['avg_time']
    for m, d in results.items():
        if m == 'autograd': continue
        mdiff = torch.max(torch.abs(d['gradient'] - base_grad)).item()
        rerr  = (torch.norm(d['gradient'] - base_grad) / torch.norm(base_grad)).item()
        speed = base_time / d['avg_time']
        save  = base_time - d['avg_time']
        pct   = save / base_time * 100
        print(f"\n{names[m]}:")
        print(f"  正确性: max {mdiff:.2e}, rel {rerr:.2e} {'✓' if mdiff<1e-5 else '⚠️'}")
        print(f"  性能: {speed:.2f}x，{'节省' if speed>1 else '增加'}时间 {abs(save):.2f} ms ({abs(pct):.1f}%)")

print("\n详细统计")
print("-"*40)
for m, d in results.items():
    print(f"{names[m]}: 最快 {d['min_time']:.2f} ms | 最慢 {d['max_time']:.2f} ms | 平均 {d['avg_time']:.2f} ms | σ={d['std_time']:.2f} ms")

if results:
    fastest = min(results, key=lambda k: results[k]['avg_time'])
    print(f"\n🏆 最快方法: {names[fastest]} ({results[fastest]['avg_time']:.2f} ms)")

if device.type == 'cuda':
    mem_used = torch.cuda.memory_allocated(device) / 1024**2
    mem_rsvd = torch.cuda.memory_reserved(device) / 1024**2
    print(f"\nGPU内存使用: {mem_used:.2f} MB (缓存: {mem_rsvd:.2f} MB)")
else:
    print("\n运行模式: CPU")

print(f"\n问题规模: M={M}, N={N}, K={K}")
print(f"总参数量(估算): M + 2*M*N + 2*sum(C(K,i), i=2..5) + 2*K + 2 = "
      f"{M + 2*M*N + 2*(len(idx2)+len(idx3)+len(idx4)+len(idx5)) + 2*K + 2}")
