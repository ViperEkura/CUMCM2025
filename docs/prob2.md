

$$
\begin{aligned}
& \min P = \sum_{i=1}^{n} w_i g(T_i) \\
& \text{s.t.} 
\begin{cases} 
& (1)\quad b_0 < b_1 < b_2 < \cdots < b_{n-1} < b_n \\
& (2)\quad 10 \leq T_i \leq 25, \quad \forall i = 1,2,\ldots,n \\
& (3)\quad N_i > 25, \quad \forall i = 1,2,\ldots,n \\
& (4)\quad P(y > 0.04| T_i) > 0.9 \\
& (5)\quad w_i = \frac{N_i}{N}, \quad \forall i = 1,2,\ldots,n \\
& (6)\quad g(T_i) = T_i - 10
\quad \forall i = 1,2,\ldots,n \\
& (7)\quad b_0 = BMI_{min},\quad b_n = BMI_{max} \\
\end{cases}
\end{aligned}
$$

### 说明与补充：

- **决策变量**：  
  - 区间分割点 $b_i$（决定 BMI 的分段区间）  
- **参数**：  
  - $n$：总区间数（给定）  
  - $BMI_{\min},BMI_{\max}$：BMI 的取值范围（给定）
  - $N_i$ : 第 $i$ 个区间内部的样本数（给定）
  - $N$:  总共样本数（给定）
  - $T_i$: BMI 的分段区间内选定的t检测阈值（给定） 

