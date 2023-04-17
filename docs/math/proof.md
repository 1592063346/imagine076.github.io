# 代数基本定理的四种复分析/复变函数证法

看来，我也是穿着长衫的孔乙己吧……

## 代数基本定理

任意复系数多项式

$$P(z) = a_0z^n + a_1z^{n - 1} + \cdots + a_n, ~~ a_0 \neq 0$$

在 $\mathbb C$ 中必有零点.

## 证法一：使用 Liouville 定理

!!! note "Liouville 定理"

    有界整函数必为常数.

如果 $P(z)$ 在 $\mathbb C$ 中没有零点，那么 $\displaystyle f(z) = \frac{1}{P(z)}$ 是一个整函数. 由于 $\displaystyle \lim_{z\to\infty} P(z) = \infty$，故 $\displaystyle \lim_{z\to\infty} f(z) = 0$，即 $f$ 是有界整函数，从而依 Liouville 定理，$f$ 是常数，矛盾.

## 证法二：使用辐角原理

!!! note "辐角原理"

    设 $D$ 是由 $\mathbb C$ 中有限条可求长简单闭曲线围成的区域，$f \in H(\overline{D})$ 且在 $\partial D$ 上无零点. 记 $m$ 为 $f$ 在 $D$ 中的零点个数（记重数），则
    
    $$\frac{1}{2\pi{\rm i}}\int_{\partial D} \frac{f'(z)}{f(z)} = \frac{1}{2\pi}\Delta_{\partial D}{\rm Arg}\,f(z) = m.$$

由于 $\displaystyle \lim_{z\to\infty} P(z) = \infty$，故当 $R > 0$ 足够大时，$P$ 在 $\gamma: |z| = R$ 上不存在零点，从而依辐角原理，在 $B(0, R)$ 内 $P$ 的零点个数为 $\displaystyle \frac{1}{2\pi}\Delta_{\gamma}{\rm Arg}\,P(z)$. 注意到

$$\Delta_{\gamma}{\rm Arg}\, P(z) = \Delta_{\gamma}{\rm Arg}\,z^n + \Delta_{\gamma}{\rm Arg}\left(a_0 + \frac{a_1}{z} + \cdots + \frac{a_n}{z^n}\right),$$

其中，$\Delta_{\gamma}{\rm Arg}\,z^n = 2n\pi$（因为 $0$ 显然是 $f(z) = z^n$ 的唯一零点，重数为 $n$），而当 $R$ 足够大时，$\displaystyle a_0 + \frac{a_1}{z} + \cdots + \frac{a_n}{z^n}$ 只在一个以 $a_0$ 为中心且不包含 $0$ 的邻域内取值，这给出 $\displaystyle \Delta_{\gamma}{\rm Arg}\left(a_0 + \frac{a_1}{z} + \cdots + \frac{a_n}{z^n}\right) = 0$. 因此，当 $R$ 足够大时，$\Delta_{\gamma}{\rm Arg}\, P(z) = 2n\pi$，即 $P$ 在 $B(0, R)$ 内存在零点.

## 证法三：使用 Rouché 定理

!!! note "Rouché 定理"

    设 $D$ 是区域，$f, g \in H(D)$，$\gamma$ 是 $D$ 中可求长的简单闭曲线，$\gamma$ 的内部位于 $D$ 中. 如果

    $$|f(z) - g(z)| < |f(z)|, ~~ \forall z \in \gamma,$$

    那么 $f$ 和 $g$ 在 $\gamma$ 内部的零点个数相同.

令 $f(z) = a_0z^n, g(z) = P(z)$，则 $\deg \big(f(z) - g(z)\big) \leq n-1 < \deg f(z) = n$，从而当 $R$ 足够大时，在 $|z| = R$ 上有 $|f(z) - g(z)| < |f(z)|$. 由于 $f$ 存在零点 $0$，故当 $R$ 足够大时，依 Rouché 定理，$g$ 在 $B(0, R)$ 内也存在零点.

## 证法四：使用最大模原理

!!! note "最大模原理"

    设 $f$ 是域 $D$ 中非常数的全纯函数，那么 $|f(z)|$ 不可能在 $D$ 中取到最大值.

如果 $P(z)$ 在 $\mathbb C$ 中没有零点，那么 $\displaystyle f(z) = \frac{1}{P(z)}$ 是一个整函数. 由于 $\displaystyle \lim_{z\to\infty} |P(z)| = +\infty$，故 $\displaystyle \lim_{z\to\infty} |f(z)| = 0$，从而依最大模原理，$|f| \equiv 0$（若存在 $z_0 \in \mathbb C$ 使 $|f(z_0)| > 0$，则 $|f(z)|$ 在 $\mathbb C$ 上可以取到最大值），矛盾.
