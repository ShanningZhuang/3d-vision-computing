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

1.**[Programming Assignment]** Let **p** = $\left( \frac{\pi}{4}, \frac{\pi}{6} \right)$ and **v** = $(1, 0)$. Draw the curve of $f(\gamma(t))$ on the ellipsoid’s surface.

**Hint:** You can use [open3d.geometry.TriangleMesh](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d-geometry-trianglemesh) to create various simple geometries.  
To visualize the curve in 3D space, you can show sampled 3D points from the curve, or use piece-wise cylinders to approximate the curve.
