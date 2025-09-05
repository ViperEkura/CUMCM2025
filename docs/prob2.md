

$$
\begin{aligned}
& \min P = \sum_{i=1}^{n} w_i g(t_i) \\
& \text{s.t.} 
\begin{cases} 
& (1)\quad b_0 < b_1 < b_2 < \cdots < b_{n-1} < b_n \\
& (2)\quad 10 \leq t_i \leq 25, \quad \forall i = 1,2,\ldots,n \\
& (3)\quad N_i > 25, \quad \forall i = 1,2,\ldots,n \\
& (4)\quad w_i = \frac{N_i}{N}, \quad \forall i = 1,2,\ldots,n \\
& (5)\quad g(t_i) = 
\begin{cases} 
1 & \text{if } t_i \leq 12 \\
2 & \text{if } 12 < t_i \leq 25
\end{cases}
\quad \forall i = 1,2,\ldots,n \\
& (6)\quad b_0 = BMI_{min},\quad b_n = BMI_{max} \\
\end{cases}
\end{aligned}
$$

### 说明与补充：

- **决策变量**：  
  - 区间分割点 $b_i$（决定 BMI 的分段区间）  
  - 检测阈值 $t_i$（每个区间的检测参数）  
- **参数**：  
  - $n$：总区间数（给定）  
  - $BMI_{\min},BMI_{\max}$：BMI 的取值范围（给定）
  - $N_i$ : 第 $i$ 个区间内部的样本数（给定）
  - $N$:  总共样本数（给定）

