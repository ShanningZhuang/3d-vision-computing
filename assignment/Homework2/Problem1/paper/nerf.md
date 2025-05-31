# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

**Ben Mildenhall¹⁽*⁾, Pratul P. Srinivasan¹⁽*⁾, Matthew Tancik¹⁽*⁾, Jonathan T. Barron², Ravi Ramamoorthi³, Ren Ng¹**

¹UC Berkeley, ²Google Research, ³UC San Diego

⁽*⁾ Authors contributed equally to this work.

## Abstract

We present a method that achieves state-of-the-art results for synthesizing novel views of complex scenes by optimizing an underlying continuous volumetric scene function using a sparse set of input views. Our algorithm represents a scene using a fully-connected (non-convolutional) deep network, whose input is a single continuous 5D coordinate (spatial location (x, y, z) and viewing direction (θ, φ)) and whose output is the volume density and view-dependent emitted radiance at that spatial location. We synthesize views by querying 5D coordinates along camera rays and use classic volume rendering techniques to project the output colors and densities into an image. Because volume rendering is naturally differentiable, the only input required to optimize our representation is a set of images with known camera poses. We describe how to effectively optimize neural radiance fields to render photorealistic novel views of scenes with complicated geometry and appearance, and demonstrate results that outperform prior work on neural rendering and view synthesis. View synthesis results are best viewed as videos, so we urge readers to view our supplementary video for convincing comparisons.

**Keywords:** scene representation, view synthesis, image-based rendering, volume rendering, 3D deep learning

## 1 Introduction

In this work, we address the long-standing problem of view synthesis in a new way by directly optimizing parameters of a continuous 5D scene representation to minimize the error of rendering a set of captured images.

We represent a static scene as a continuous 5D function that outputs the radiance emitted in each direction (θ, φ) at each point (x, y, z) in space, and a density at each point which acts like a differential opacity controlling how much radiance is accumulated by a ray passing through (x, y, z). Our method optimizes a deep fully-connected neural network without any convolutional layers (often referred to as a multilayer perceptron or MLP) to represent this function by regressing from a single 5D coordinate (x, y, z, θ, φ) to a single volume density and view-dependent RGB color. To render this neural radiance field (NeRF)

(cid:63) Authors contributed equally to this work.

Fig. 1: We present a method that optimizes a continuous 5D neural radiance
field representation (volume density and view-dependent color at any continuous
location) of a scene from a set of input images. We use techniques from volume
rendering to accumulate samples of this scene representation along rays to render
the scene from any viewpoint. Here, we visualize the set of 100 input views of the
synthetic Drums scene randomly captured on a surrounding hemisphere, and we
show two novel views rendered from our optimized NeRF representation.

from a particular viewpoint we: 1) march camera rays through the scene to
generate a sampled set of 3D points, 2) use those points and their corresponding
2D viewing directions as input to the neural network to produce an output
set of colors and densities, and 3) use classical volume rendering techniques to
accumulate those colors and densities into a 2D image. Because this process is
naturally differentiable, we can use gradient descent to optimize this model by
minimizing the error between each observed image and the corresponding views
rendered from our representation. Minimizing this error across multiple views
encourages the network to predict a coherent model of the scene by assigning
high volume densities and accurate colors to the locations that contain the true
underlying scene content. Figure 2 visualizes this overall pipeline.

We ﬁnd that the basic implementation of optimizing a neural radiance ﬁeld
representation for a complex scene does not converge to a suﬃciently high-
resolution representation and is ineﬃcient in the required number of samples per
camera ray. We address these issues by transforming input 5D coordinates with
a positional encoding that enables the MLP to represent higher frequency func-
tions, and we propose a hierarchical sampling procedure to reduce the number of
queries required to adequately sample this high-frequency scene representation.
Our approach inherits the beneﬁts of volumetric representations: both can
represent complex real-world geometry and appearance and are well suited for
gradient-based optimization using projected images. Crucially, our method over-
comes the prohibitive storage costs of discretized voxel grids when modeling
complex scenes at high-resolutions. In summary, our technical contributions are:
– An approach for representing continuous scenes with complex geometry and
materials as 5D neural radiance ﬁelds, parameterized as basic MLP networks.
– A diﬀerentiable rendering procedure based on classical volume rendering tech-
niques, which we use to optimize these representations from standard RGB
images. This includes a hierarchical sampling strategy to allocate the MLP's
capacity towards space with visible scene content.

– A positional encoding to map each input 5D coordinate into a higher dimen-
sional space, which enables us to successfully optimize neural radiance ﬁelds
to represent high-frequency scene content.

We demonstrate that our resulting neural radiance ﬁeld method quantitatively
and qualitatively outperforms state-of-the-art view synthesis methods, including
works that ﬁt neural 3D representations to scenes as well as works that train deep
convolutional networks to predict sampled volumetric representations. As far as
we know, this paper presents the ﬁrst continuous neural scene representation
that is able to render high-resolution photorealistic novel views of real objects
and scenes from RGB images captured in natural settings.

2 Related Work

A promising recent direction in computer vision is encoding objects and scenes
in the weights of an MLP that directly maps from a 3D spatial location to
an implicit representation of the shape, such as the signed distance [6] at that
location. However, these methods have so far been unable to reproduce realistic
scenes with complex geometry with the same ﬁdelity as techniques that represent
scenes using discrete representations such as triangle meshes or voxel grids. In
this section, we review these two lines of work and contrast them with our
approach, which enhances the capabilities of neural scene representations to
produce state-of-the-art results for rendering complex realistic scenes.

A similar approach of using MLPs to map from low-dimensional coordinates
to colors has also been used for representing other graphics functions such as im-
ages [44], textured materials [12,31,36,37], and indirect illumination values [38].

**Neural 3D shape representations** Recent work has investigated the implicit representation of continuous 3D shapes as level sets by optimizing deep
networks that map xyz coordinates to signed distance functions [15,32] or occu-
pancy ﬁelds [11,27]. However, these models are limited by their requirement of
access to ground truth 3D geometry, typically obtained from synthetic 3D shape
datasets such as ShapeNet [3]. Subsequent work has relaxed this requirement of
ground truth 3D shapes by formulating diﬀerentiable rendering functions that
allow neural implicit shape representations to be optimized using only 2D im-
ages. Niemeyer et al.
[29] represent surfaces as 3D occupancy ﬁelds and use a
numerical method to ﬁnd the surface intersection for each ray, then calculate an
exact derivative using implicit diﬀerentiation. Each ray intersection location is
provided as the input to a neural 3D texture ﬁeld that predicts a diﬀuse color for
that point. Sitzmann et al.
[42] use a less direct neural 3D representation that
simply outputs a feature vector and RGB color at each continuous 3D coordinate,
and propose a diﬀerentiable rendering function consisting of a recurrent neural
network that marches along each ray to decide where the surface is located.

Though these techniques can potentially represent complicated and high-
resolution geometry, they have so far been limited to simple shapes with low
geometric complexity, resulting in oversmoothed renderings. We show that an al-
ternate strategy of optimizing networks to encode 5D radiance ﬁelds (3D volumes

with 2D view-dependent appearance) can represent higher-resolution geometry
and appearance to render photorealistic novel views of complex scenes.

**View synthesis and image-based rendering** Given a dense sampling of
views, photorealistic novel views can be reconstructed by simple light ﬁeld sam-
ple interpolation techniques [21,5,7]. For novel view synthesis with sparser view
sampling, the computer vision and graphics communities have made signiﬁcant
progress by predicting traditional geometry and appearance representations from
observed images. One popular class of approaches uses mesh-based representa-
tions of scenes with either diﬀuse [48] or view-dependent [2,8,49] appearance.
Diﬀerentiable rasterizers [4,10,23,25] or pathtracers [22,30] can directly optimize
mesh representations to reproduce a set of input images using gradient descent.
However, gradient-based mesh optimization based on image reprojection is often
diﬃcult, likely because of local minima or poor conditioning of the loss land-
scape. Furthermore, this strategy requires a template mesh with ﬁxed topology
to be provided as an initialization before optimization [22], which is typically
unavailable for unconstrained real-world scenes.

Another class of methods use volumetric representations to address the task
of high-quality photorealistic view synthesis from a set of input RGB images.
Volumetric approaches are able to realistically represent complex shapes and
materials, are well-suited for gradient-based optimization, and tend to produce
less visually distracting artifacts than mesh-based methods. Early volumetric
approaches used observed images to directly color voxel grids [19,40,45]. More
recently, several methods [9,13,17,28,33,43,46,52] have used large datasets of mul-
tiple scenes to train deep networks that predict a sampled volumetric represen-
tation from a set of input images, and then use either alpha-compositing [34] or
learned compositing along rays to render novel views at test time. Other works
have optimized a combination of convolutional networks (CNNs) and sampled
voxel grids for each speciﬁc scene, such that the CNN can compensate for dis-
cretization artifacts from low resolution voxel grids [41] or allow the predicted
voxel grids to vary based on input time or animation controls [24]. While these
volumetric techniques have achieved impressive results for novel view synthe-
sis, their ability to scale to higher resolution imagery is fundamentally limited
by poor time and space complexity due to their discrete sampling — rendering
higher resolution images requires a ﬁner sampling of 3D space. We circumvent
this problem by instead encoding a continuous volume within the parameters
of a deep fully-connected neural network, which not only produces signiﬁcantly
higher quality renderings than prior volumetric approaches, but also requires
just a fraction of the storage cost of those sampled volumetric representations.

3 Neural Radiance Field Scene Representation

We represent a continuous scene as a 5D vector-valued function whose input is
a 3D location x = (x, y, z) and 2D viewing direction (θ, φ), and whose output
is an emitted color c = (r, g, b) and volume density σ. In practice, we express

direction as a 3D Cartesian unit vector d. We approximate this continuous 5D
scene representation with an MLP network F_Θ : (x, d) → (c, σ) and optimize its
weights Θ to map from each input 5D coordinate to its corresponding volume
density and directional emitted color.

We encourage the representation to be multiview consistent by restricting
the network to predict the volume density σ as a function of only the location
x, while allowing the RGB color c to be predicted as a function of both location
and viewing direction. To accomplish this, the MLP F_Θ first processes the input
3D coordinate x with 8 fully-connected layers (using ReLU activations and 256
channels per layer), and outputs σ and a 256-dimensional feature vector. This
feature vector is then concatenated with the camera ray's viewing direction and
passed to one additional fully-connected layer (using a ReLU activation and 128
channels) that output the view-dependent RGB color.

See Fig. 3 for an example of how our method uses the input viewing direction
to represent non-Lambertian eﬀects. As shown in Fig. 4, a model trained without
view dependence (only x as input) has diﬃculty representing specularities.

4 Volume Rendering with Radiance Fields

Our 5D neural radiance field represents a scene as the volume density and di-
rectional emitted radiance at any point in space. We render the color of any ray
passing through the scene using principles from classical volume rendering [16].
The volume density σ(x) can be interpreted as the diﬀerential probability of a
ray terminating at an inﬁnitesimal particle at location x. The expected color
C(r) of camera ray r(t) = o + td with near and far bounds t_n and t_f is:

C(r) = ∫[t_n to t_f] T(t)σ(r(t))c(r(t), d)dt, where T(t) = exp(-∫[t_n to t] σ(r(s))ds). (1)

The function T(t) denotes the accumulated transmittance along the ray from t_n to t, i.e., the probability that the ray travels from t_n to t without hitting any other particle. Rendering a view from our continuous neural radiance field requires estimating this integral C(r) for a camera ray traced through each pixel of the desired virtual camera.

We numerically estimate this continuous integral using quadrature. Deterministic quadrature, which is typically used for rendering discretized voxel grids, would effectively limit our representation's resolution because the MLP would only be queried at a fixed discrete set of locations. Instead, we use a stratified sampling approach where we partition [t_n, t_f] into N evenly-spaced bins and then draw one sample uniformly at random from within each bin:

t_i ∼ U[t_n + (i-1)/N (t_f - t_n), t_n + i/N (t_f - t_n)]. (2)

Although we use a discrete set of samples to estimate the integral, stratified sampling enables us to represent a continuous scene representation because it results in the MLP being evaluated at continuous positions over the course of optimization. We use these samples to estimate C(r) with the quadrature rule discussed in the volume rendering review by Max [26]:

Ĉ(r) = Σ[i=1 to N] T_i(1 - exp(-σ_i δ_i))c_i, where T_i = exp(-Σ[j=1 to i-1] σ_j δ_j), (3)

where δ_i = t_{i+1} - t_i is the distance between adjacent samples. This function for calculating Ĉ(r) from the set of (c_i, σ_i) values is trivially differentiable and reduces to traditional alpha compositing with alpha values α_i = 1 - exp(-σ_i δ_i).

## 5 Optimizing a Neural Radiance Field

In the previous section we have described the core components necessary for modeling a scene as a neural radiance field and rendering novel views from this representation. However, we observe that these components are not sufficient for achieving state-of-the-art quality, as demonstrated in Section 6.4). We introduce two improvements to enable representing high-resolution complex scenes. The first is a positional encoding of the input coordinates that assists the MLP in representing high-frequency functions, and the second is a hierarchical sampling procedure that allows us to efficiently sample this high-frequency representation.

### 5.1 Positional encoding

Despite the fact that neural networks are universal function approximators [14], we found that having the network F_Θ directly operate on xyzθφ input coordinates results in renderings that perform poorly at representing high-frequency variation in color and geometry. This is consistent with recent work by Rahaman et al. [35], which shows that deep networks are biased towards learning lower frequency functions. They additionally show that mapping the inputs to a higher dimensional space using high frequency functions before passing them to the network enables better fitting of data that contains high frequency variation.

We leverage these findings in the context of neural scene representations, and show that reformulating F_Θ as a composition of two functions F_Θ = F'_Θ ∘ γ, one learned and one not, significantly improves performance (see Fig. 4 and Table 2). Here γ is a mapping from R into a higher dimensional space R^{2L}, and F'_Θ is still simply a regular MLP. Formally, the encoding function we use is:

γ(p) = (sin(2^0πp), cos(2^0πp), ⋯, sin(2^{L-1}πp), cos(2^{L-1}πp)). (4)

This function γ(·) is applied separately to each of the three coordinate values in x (which are normalized to lie in [-1, 1]) and to the three components of the Cartesian viewing direction unit vector d (which by construction lie in [-1, 1]). In our experiments, we set L = 10 for γ(x) and L = 4 for γ(d).

A similar mapping is used in the popular Transformer architecture [47], where it is referred to as a positional encoding. However, Transformers use it for a different goal of providing the discrete positions of tokens in a sequence as input to an architecture that does not contain any notion of order. In contrast, we use these functions to map continuous input coordinates into a higher dimensional space to enable our MLP to more easily approximate a higher frequency function. Concurrent work on a related problem of modeling 3D protein structure from projections [51] also utilizes a similar input coordinate mapping.

### 5.2 Hierarchical volume sampling

Our rendering strategy of densely evaluating the neural radiance field network at N query points along each camera ray is inefficient: free space and occluded regions that do not contribute to the rendered image are still sampled repeatedly. We draw inspiration from early work in volume rendering [20] and propose a hierarchical representation that increases rendering efficiency by allocating samples proportionally to their expected effect on the final rendering.

Instead of just using a single network to represent the scene, we simultaneously optimize two networks: one "coarse" and one "fine". We first sample a set of N_c locations using stratified sampling, and evaluate the "coarse" network at these locations as described in Eqns. 2 and 3. Given the output of this "coarse" network, we then produce a more informed sampling of points along each ray where samples are biased towards the relevant parts of the volume. To do this, we first rewrite the alpha composited color from the coarse network Ĉ_c(r) in Eqn. 3 as a weighted sum of all sampled colors c_i along the ray:

Ĉ_c(r) = Σ[i=1 to N_c] w_i c_i, w_i = T_i(1 - exp(-σ_i δ_i)). (5)

Normalizing these weights as ŵ_i = w_i/(Σ[j=1 to N_c] w_j) produces a piecewise-constant PDF along the ray. We sample a second set of N_f locations from this distribution using inverse transform sampling, evaluate our "fine" network at the union of the first and second set of samples, and compute the final rendered color of the ray Ĉ_f(r) using Eqn. 3 but using all N_c + N_f samples. This procedure allocates more samples to regions we expect to contain visible content. This addresses a similar goal as importance sampling, but we use the sampled values as a nonuniform discretization of the whole integration domain rather than treating each sample as an independent probabilistic estimate of the entire integral.

### 5.3 Implementation details

We optimize a separate neural continuous volume representation network for each scene. This requires only a dataset of captured RGB images of the scene,

the corresponding camera poses and intrinsic parameters, and scene bounds
(we use ground truth camera poses, intrinsics, and bounds for synthetic data,
and use the COLMAP structure-from-motion package [39] to estimate these
parameters for real data). At each optimization iteration, we randomly sample
a batch of camera rays from the set of all pixels in the dataset, and then follow
the hierarchical sampling described in Sec. 5.2 to query N_c samples from the
coarse network and N_c + N_f samples from the fine network. We then use the
volume rendering procedure described in Sec. 4 to render the color of each ray
from both sets of samples. Our loss is simply the total squared error between
the rendered and true pixel colors for both the coarse and fine renderings:

L =

(cid:88)

r∈R

(cid:20)(cid:13)
(cid:13)
2
ˆCc(r) − C(r)
(cid:13)
(cid:13)
(cid:13)
(cid:13)
2

+

(cid:13)
ˆCf (r) − C(r)
(cid:13)
(cid:13)

(cid:13)
2
(cid:13)
(cid:13)
2

(cid:21)

(6)

where R is the set of rays in each batch, and C(r), ˆCc(r), and ˆCf (r) are the
ground truth, coarse volume predicted, and fine volume predicted RGB colors
for ray r respectively. Note that even though the final rendering comes from
ˆCf (r), we also minimize the loss of ˆCc(r) so that the weight distribution from
the coarse network can be used to allocate samples in the fine network.

In our experiments, we use a batch size of 4096 rays, each sampled at N_c = 64
coordinates in the coarse volume and N_f = 128 additional coordinates in the
fine volume. We use the Adam optimizer [18] with a learning rate that begins at
5 × 10^-4 and decays exponentially to 5 × 10^-5 over the course of optimization
(other Adam hyperparameters are left at default values of β1 = 0.9, β2 = 0.999,
and (cid:15) = 10^-7). The optimization for a single scene typically take around 100–
300k iterations to converge on a single NVIDIA V100 GPU (about 1–2 days).

6 Results

We quantitatively (Tables 1) and qualitatively (Figs. 8 and 6) show that our
method outperforms prior work, and provide extensive ablation studies to vali-
date our design choices (Table 2). We urge the reader to view our supplementary
video to better appreciate our method's signiﬁcant improvement over baseline
methods when rendering smooth paths of novel views.

### 6.1 Datasets

**Synthetic renderings of objects** We first show experimental results on two
datasets of synthetic renderings of objects (Table 1, "Diffuse Synthetic 360°" and
"Realistic Synthetic 360°"). The DeepVoxels [41] dataset contains four Lambert-
tian objects with simple geometry. Each object is rendered at 512 × 512 pixels
from viewpoints sampled on the upper hemisphere (479 as input and 1000 for
testing). We additionally generate our own dataset containing pathtraced images
of eight objects that exhibit complicated geometry and realistic non-Lambertian
materials. Six are rendered from viewpoints sampled on the upper hemisphere,
and two are rendered from viewpoints sampled on a full sphere. We render 100
views of each scene as input and 200 for testing, all at 800 × 800 pixels.

**Real images of complex scenes** We show results on complex real-world
scenes captured with roughly forward-facing images (Table 1, "Real Forward-
Facing"). This dataset consists of 8 scenes captured with a handheld cellphone
(5 taken from the LLFF paper and 3 that we capture), captured with 20 to 62
images, and hold out 1/8 of these for the test set. All images are 1008×756 pixels.

### 6.2 Comparisons

To evaluate our model we compare against current top-performing techniques
for view synthesis, detailed below. All methods use the same set of input views
to train a separate network for each scene except Local Light Field Fusion [28],
which trains a single 3D convolutional network on a large dataset, then uses the
same trained network to process input images of new scenes at test time.

**Neural Volumes (NV) [24]** synthesizes novel views of objects that lie entirely within a bounded volume in front of a distinct background (which must
be separately captured without the object of interest). It optimizes a deep 3D
convolutional network to predict a discretized RGBα voxel grid with 128³ sam-
ples as well as a 3D warp grid with 32³ samples. The algorithm renders novel
views by marching camera rays through the warped voxel grid.

**Scene Representation Networks (SRN) [42]** represent a continuous scene
as an opaque surface, implicitly deﬁned by a MLP that maps each (x, y, z) co-
ordinate to a feature vector. They train a recurrent neural network to march
along a ray through the scene representation by using the feature vector at any
3D coordinate to predict the next step size along the ray. The feature vector
from the ﬁnal step is decoded into a single color for that point on the surface.
Note that SRN is a better-performing followup to DeepVoxels [41] by the same
authors, which is why we do not include comparisons to DeepVoxels.

**Local Light Field Fusion (LLFF) [28]** LLFF is designed for producing pho-
torealistic novel views for well-sampled forward facing scenes. It uses a trained 3D
convolutional network to directly predict a discretized frustum-sampled RGBα
grid (multiplane image or MPI [52]) for each input view, then renders novel
views by alpha compositing and blending nearby MPIs into the novel viewpoint.

### 6.3 Discussion

We thoroughly outperform both baselines that also optimize a separate network
per scene (NV and SRN) in all scenarios. Furthermore, we produce qualitatively
and quantitatively superior renderings compared to LLFF (across all except one
metric) while using only their input images as our entire training set.

The SRN method produces heavily smoothed geometry and texture, and its
representational power for view synthesis is limited by selecting only a single
depth and color per camera ray. The NV baseline is able to capture reasonably
detailed volumetric geometry and appearance, but its use of an underlying ex-
plicit 128³ voxel grid prevents it from scaling to represent ﬁne details at high
resolutions. LLFF speciﬁcally provides a "sampling guideline" to not exceed 64
pixels of disparity between input views, so it frequently fails to estimate cor-
rect geometry in the synthetic datasets which contain up to 400-500 pixels of
disparity between views. Additionally, LLFF blends between diﬀerent scene rep-
resentations for rendering diﬀerent views, resulting in perceptually-distracting
inconsistency as is apparent in our supplementary video.

The biggest practical tradeoﬀs between these methods are time versus space.
All compared single scene methods take at least 12 hours to train per scene. In
contrast, LLFF can process a small input dataset in under 10 minutes. However,
LLFF produces a large 3D voxel grid for every input image, resulting in enor-
mous storage requirements (over 15GB for one "Realistic Synthetic" scene). Our
method requires only 5 MB for the network weights (a relative compression of
3000× compared to LLFF), which is even less memory than the input images
alone for a single scene from any of our datasets.

### 6.4 Ablation studies

We validate our algorithm's design choices and parameters with an extensive
ablation study in Table 2. We present results on our "Realistic Synthetic 360°"
scenes. Row 9 shows our complete model as a point of reference. Row 1 shows
a minimalist version of our model without positional encoding (PE), view-
dependence (VD), or hierarchical sampling (H). In rows 2–4 we remove these
three components one at a time from the full model, observing that positional
encoding (row 2) and view-dependence (row 3) provide the largest quantitative
beneﬁt followed by hierarchical sampling (row 4). Rows 5–6 show how our per-
formance decreases as the number of input images is reduced. Note that our
method's performance using only 25 input images still exceeds NV, SRN, and
LLFF across all metrics when they are provided with 100 images (see supple-
mentary material). In rows 7–8 we validate our choice of the maximum frequency

L used in our positional encoding for x (the maximum frequency used for d is
scaled proportionally). Only using 5 frequencies reduces performance, but in-
creasing the number of frequencies from 10 to 15 does not improve performance.
We believe the beneﬁt of increasing L is limited once 2L exceeds the maximum
frequency present in the sampled input images (roughly 1024 in our data).

## 7 Conclusion

Our work directly addresses deficiencies of prior work that uses MLPs to represent objects and scenes as continuous functions. We demonstrate that representing scenes as 5D neural radiance fields (an MLP that outputs volume density and view-dependent emitted radiance as a function of 3D location and 2D viewing direction) produces better renderings than the previously-dominant approach of training deep convolutional networks to output discretized voxel representations.

Although we have proposed a hierarchical sampling strategy to make rendering more sample-efficient (for both training and testing), there is still much more progress to be made in investigating techniques to efficiently optimize and render neural radiance fields. Another direction for future work is interpretability: sampled representations such as voxel grids and meshes admit reasoning about the expected quality of rendered views and failure modes, but it is unclear how to analyze these issues when we encode scenes in the weights of a deep neural network. We believe that this work makes progress towards a graphics pipeline based on real world imagery, where complex scenes could be composed of neural radiance fields optimized from images of actual objects and scenes.

## Acknowledgements

We thank Kevin Cao, Guowei Frank Yang, and Nithin Raghavan for comments and discussions. RR acknowledges funding from ONR grants N000141712687 and N000142012529 and the Ronald L. Graham Chair. BM is funded by a Hertz Foundation Fellowship, and MT is funded by an NSF Graduate Fellowship. Google provided a generous donation of cloud compute credits through the BAIR Commons program. We thank the following Blend Swap users for the models used in our realistic synthetic dataset: gregzaal (ship), 1DInc (chair), bryanajones (drums), Herberhold (ficus), erickfree (hotdog), Heinzelnisse (lego), elbrujodelatribu (materials), and up3d.de (mic).

## References

1. Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G.S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., Zheng, X.: TensorFlow: Large-scale machine learning on heterogeneous systems (2015)


