# Evaluating Physics-Informed Neural Networks (PINNs) for One-Dimensional Pseudo-Parabolic PDEs

## Introduction
In recent years, neuronal networks for approximating the solution of 
partial differential equations have become an active area of research. 
A substantial impact on the research was the introduction
of PINNs by Raissi et. al., cf. [2]. 
In [1], Wang et. al. proposed training strategies to improve the results for computing 
approximate solutions for parabolic equations by PINNs. In the presented project we evaluate 
whether the training strategies introduced in [1] can also be applied
to one-dimensional pseudo-parabolic partial differential equations. In our experiments
we focus on the following class of pseudo-parabolic equations: Given a bounded interval 
$(\alpha, \beta) \subset \mathbb{R}$, continuous functions $a, c \in C([\alpha, \beta])$ and 
a source function $F:[0,T] \to C([\alpha, \beta])$, which is also continuous on $[0,T]$,
find a function $u:[0,T] \to \mathbb{R}$ that satisfies

```math
-u_{xxt} + au_t - u_{xx} + cu = F \quad \text{on } (a,b) \times (0,T] \\
u(x,0) = u_0(x) \quad \text{on } \overline{\Omega} \\
u(x,t) = 0 \quad \text{for } (x,t) \in \{\alpha, \beta\} \times [0,T], 
```

where $u_t$ and $u_x$ denote the parital derivatives with respect to the time
variable $t$ and the space variable $x$, respetively. We assume that $a \geq 0$ on 
$\overline{\Omega}$ and $u_0 \in C^1([\alpha, \beta])$. Under these assumptions, 
our pseudo-parabolic equation possesses a unique solution $u$.




### Literature
[1] Sifan Wang, Shyam Sankaran, Hanwen Wang, Paris Perdikaris,
    An Expert's Guide to Training Physics-Informed Neuronal Networks, 
    2023, https://arxiv.org/abs/2308.08468
[2] 