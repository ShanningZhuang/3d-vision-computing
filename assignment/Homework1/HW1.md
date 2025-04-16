### **Problem 1**

In this problem, you will do a bit of calculus to see how the operators we introduced in the class work on a simple ellipsoid. Consider the map  
$f : \mathbb{R}^2 \rightarrow \mathbb{R}^3$ defined by:

$$
f(u,v) = \begin{bmatrix}
a \cos u \sin v \\
b \sin u \sin v \\
c \cos v
\end{bmatrix},
\quad \text{where } -\pi \leq u \leq \pi,\ 0 \leq v \leq \pi.
$$

The function $f$ maps the 2D domain to an ellipsoid.

Let $a = 1$, $b = 1$, $c = \frac{1}{2}$. Let $\mathbf{p} = (u, v)$ be a point in the domain of $f$, and let  
$\gamma : (-1, 1) \rightarrow \mathbb{R}^2$ be a curve with $\gamma(0) = \mathbf{p}$ and $\gamma'(t) = \mathbf{v}$.

### 1. **[Programming Assignment]** Let **p** = $\left( \frac{\pi}{4}, \frac{\pi}{6} \right)$ and **v** = $(1, 0)$. Draw the curve of $f(\gamma(t))$ on the ellipsoid’s surface.

**Hint:** You can use [open3d.geometry.TriangleMesh](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d-geometry-trianglemesh) to create various simple geometries.  
To visualize the curve in 3D space, you can show sampled 3D points from the curve, or use piece-wise cylinders to approximate the curve.

### 2. **Differential map:**

The differential of a function $f$ at a point **p** is a linear map. Equivalent to how we define the differential in class, we can also define it by the gradient of the curve w.r.t. the curve parameter:  
$$
D f_{\mathbf{p}}(\mathbf{v}) = \left. f(\gamma(t))' \right|_{t=0}
$$

**(a)** What is $D f_{\mathbf{p}}$? Express it as a matrix.

**(b)** Describe the geometric meaning of $D f_{\mathbf{p}}$.

**(c)** **[Programming Assignment]** Draw $D f_{\mathbf{p}}(\mathbf{v})$ on the ellipsoid when  
$\mathbf{p} = \left( \frac{\pi}{4}, \frac{\pi}{6} \right)$ and $\mathbf{v} = (1, 0)$.

**(d)** What is the normal vector of the tangent plane at **p**?

**(e)** **[Programming Assignment]** Give a group of orthonormal bases of the tangent space at $f(\mathbf{p})$ when  
$\mathbf{p} = \left( \frac{\pi}{4}, \frac{\pi}{6} \right)$, and draw it on the ellipsoid.

### 3. **Normal:**

Given $\mathbf{p} = \left( \frac{\pi}{4}, \frac{\pi}{6} \right)$ and $\mathbf{v} = (1, 0)$.  
For simplicity, let $g_{\mathbf{v}}(t) = f(\gamma(t))$ denote the curve which passes through $\mathbf{p}$ at $t = 0$.

**(a)** What is the arc length $s(t)$ as the point moves from $g_{\mathbf{v}}(0)$ to $g_{\mathbf{v}}(t)$?

**(b)** Give the arc-length parameterization $h_{\mathbf{v}}(s)$ of the curve.

**(c)** What is the normal vector of the curve at a point $h_{\mathbf{v}}(s)$?  
*Hint:* Use $h_{\mathbf{v}}(s)$ to derive the normal.

> **Note:** $h_{\mathbf{v}}(0)$ is the point $\mathbf{p}$.  
> If you compare the curve normal you get in 3(c) with the surface normal in 2(d) at $s = 0$,  
> you can find that they are different.

### 4. **Curvature:**

In 2(d), you have computed the normal at **p**. Denote this normal as $N_{\mathbf{p}}$.

**(a)** Compute the differential of the normal $DN_{\mathbf{p}}$, and express it as a matrix.  
*Hint:* You can use WolframAlpha to compute complicated derivatives that you don’t want to compute by hand.

**(b)** Find the eigenvectors of the shape operator at **p**.  
*Hint:* You can show the shape operator is diagonal (What does it tell you about the eigenvectors?).

**(c)** **[Programming Assignment]** Draw the two principal curvature directions in the tangent plane of the ellipsoid at  
$\mathbf{p} = \left( \frac{\pi}{4}, \frac{\pi}{6} \right)$.

**(d)** Compute the Gaussian curvature of the surface $f$ at  
$\mathbf{p} = \left( \frac{\pi}{4}, \frac{\pi}{6} \right)$, and demonstrate that ellipsoid $f$ doesn’t show isometric invariance with any spherical surface.