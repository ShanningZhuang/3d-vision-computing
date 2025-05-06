## Lecture 1

I. **Overall Overview**

* **Topic:** Introduction to 3D Visual Computing—scope, course structure, and fundamental geometry.&#x20;
* **Learning Objectives:**

  1. Understand the breadth of 3D visual computing, from data acquisition and representation through processing, synthesis, and analysis.
  2. Learn course logistics, prerequisites, grading policies, and available resources.
  3. Master the mathematical foundations of curves and surfaces (parametrization, tangent, normal, curvature).&#x20;
* **Core Thread:** This lecture defines 3D visual computing as the intersection of graphics, vision, and shape processing; lays out the roadmap and policies; then grounds students in the differential‑geometry basics of curves and surfaces that underpin all later topics.&#x20;

II. **Section 1 (Slides 1–5)**

1. **Summary:** Instructor introduction; course agenda (overview of 3D visual computing, syllabus, logistics, and first technical topic); and a formal definition of “3D visual computing” as the umbrella for CS disciplines handling 3D data.&#x20;
2. **Connection to Previous:** As the opening, it establishes context, key terms, and expectations for the entire course.
3. **Outlook for Next:** We’ll see concrete examples of how omnipresent 3D data is in real applications.

III. **Section 2 (Slides 6–12)**

1. **Summary:** “3D is Everywhere” through real‑world case studies (Zillow, Waymo, Oculus, Unity, etc.); and an outline of core subfields—representation, acquisition, synthesis, processing & analysis, rendering.&#x20;
2. **Connection to Previous:** Grounds the abstract definition in tangible applications, motivating why each subdomain matters.
3. **Outlook for Next:** Motivates why deep learning has rapidly become central to modern 3D workflows.

IV. **Section 3 (Slides 13–19)**

1. **Summary:** Argues for a learning‑based focus: black‑box models leverage data/experience to infer complex 3D structures; highlights neural representations like IGR and NeRF that facilitate shape recovery and rendering.&#x20;
2. **Connection to Previous:** Bridges from traditional geometry topics to data‑driven, ML‑powered methods that will recur throughout the course.
3. **Outlook for Next:** Presents the detailed course syllabus, logistics, and policies.

V. **Section 4 (Slides 20–27)**

1. **Summary:** Full course roadmap (modules on geometry, representations, reconstruction, analysis), technical prerequisites, grading breakdown, collaboration rules, and recommended resources.&#x20;
2. **Connection to Previous:** Translates motivation into an actionable plan—what you’ll learn, how you’ll be assessed, and where to find help.
3. **Outlook for Next:** Launches into the first technical module—curves and surfaces.

VI. **Section 5 (Slides 28–end)**

1. **Summary:** Core geometry: distinctions among explicit, implicit, and parametric curves γ(t); tangent γ′(t) (direction+speed); smooth 1D manifolds; reparameterization by arc length; moving frame (T, N); curvature κ(s)=‖T′(s)‖; parameterized surfaces f(u,v) and local bending via principal curvatures, Gaussian K=κ₁κ₂ and mean H=(κ₁+κ₂)/2.&#x20;
2. **Connection to Previous:** Provides the differential‑geometry toolkit needed before tackling 3D representations and learning‑based algorithms.
3. **Outlook for Next:** Next lecture will delve into concrete 3D data formats—point clouds, meshes, and implicit fields.

III. **Core Points & Memory Aids**

* **3D Visual Computing:** All CS fields handling 3D data (models, point clouds, etc.). (Aid: “3D VC = 3 Domains: Graphics, Vision, Shape”)&#x20;
* **Parametric Curve γ(t):** Maps parameter t to points in space. (Aid: “γ → ‘Geometric GPS’”)&#x20;
* **Tangent γ′(t):** Derivative of γ(t) giving instantaneous direction & speed. (Aid: “T for ‘Touch’—the direction you ‘touch’ the curve”)&#x20;
* **Curvature κ(s):** ‖T′(s)‖, measures how sharply the curve bends. (Aid: “κ flipped ‘s’—curvature along ‘s’”)&#x20;
* **Parameterized Surface f(u,v):** 2D→3D mapping (e.g., saddle). (Aid: “u,v ‘unfold’ the surface”)&#x20;
* **Gaussian Curvature K=κ₁κ₂:** Product of principal curvatures. (Aid: “G for ‘Grid’—multiplies two bending directions”)&#x20;
* **Mean Curvature H=(κ₁+κ₂)/2:** Average bending. (Aid: “H for ‘Half’—mean of two curvatures”)&#x20;

IV. **Pre-Study Suggestions**

1. **Parametric Curves:** Manually derive γ(t)=(cos t, sin t), then compute γ′(t) and plot in Python/NumPy.
2. **Vector‑Valued Derivatives:** Refresh multivariable calculus—focus on derivatives of vector functions and arc‑length parameterization.
3. **Python Practice:** Write a short script to sample a parametric curve and numerically approximate its tangent and curvature.

V. **Review Questions/Practice**

1. **Surface Curvature:** Derive Gaussian and mean curvature for the saddle f(u,v)=(u,v,u²−v²).
2. **Numerical Implementation:** Implement a Python function that takes any γ(t), computes γ′(t) and κ(t), and visualizes them.
3. **Method Comparison:** List pros & cons of analytical vs learning‑based single‑image 3D reconstruction.

## Lecture 2

I. **Overall Overview**

* **Topic:** 3D Representations in 3D Visual Computing: mesh, point‑cloud, and implicit forms.&#x20;
* **Learning Objectives:**

  1. Understand the main categories of 3D geometry representations—rasterized (multiview, depth, volumetric), mesh, point‑cloud, and implicit—and their trade‑offs.
  2. Learn key data structures and algorithms for storage, sampling, and curvature/normal computation in each representation.
* **Core Thread:** This lecture surveys how 3D shapes are captured, stored, and processed: starting from rasterized captures (multiview, depth, voxels), moving through piecewise‑linear meshes and unstructured point clouds, and concluding with continuous implicit descriptions and their interconversion.&#x20;

II. **Section Summaries**

* **Section 1 (Slides 1–5): Introduction & Recap**

  1. **Summary:** Homework announcement; recap of Lecture 1’s parametric curves/surfaces, tangents, normals, curvature; and presentation of today’s focus with an outline of representations.&#x20;
  2. **Connection to Previous:** Reasserts the differential‑geometry foundation (parametrizations and local shape measures) before exploring discrete and functional representations.
  3. **Outlook for Next:** Sets up concrete data‑acquisition and storage paradigms.

* **Section 2 (Slides 6–12): Rasterized 3D Representations**

  1. **Summary:**

     * **Multiview:** no explicit geometry, viewpoint‑dependent, reconstruction challenges.
     * **Depth Maps:** partial 3D data, require camera intrinsics, resolution/object‑property trade‑offs.
     * **Volumetric Grids:** uniform voxel arrays with high cost and artifacts; sparsity as remedy.&#x20;
  2. **Connection to Previous:** Transitions from smooth parametrizations to sampled measurements of real scenes.
  3. **Outlook for Next:** Introduces mesh as the canonical surface representation.

* **Section 3 (Slides 13–52): Mesh Representations & Curvature**

  1. **Summary:**

     * **Definitions:** polygons, triangulations, manifold vs non‑manifold.
     * **Data Structures:** triangle lists (STL), indexed face sets (OBJ/OFF), orientation conventions.
     * **Quality:** mesh cleaning/remeshing, curvature challenges on flat triangles.
     * **Curvature Estimation:** differential maps, shape operator, and Rusinkiewicz’s least‑squares method.&#x20;
  2. **Connection to Previous:** Builds on raster volumes by extracting explicit surface facets from sampled data.
  3. **Outlook for Next:** Prepares to remove connectivity and consider raw point sets.

* **Section 4 (Slides 53–64): Point‑Cloud Representations & Sampling**

  1. **Summary:**

     * **Definition:** unordered (x,y,z) sets (“surfels” with normals).
     * **Acquisition:** LIDAR, Kinect, stereo—noise, occlusion, registration issues.
     * **Sampling:** uniform, farthest‑point, voxel downsampling, application‑driven strategies.
     * **Normal Estimation:** PCA over local neighborhoods (smallest eigenvector).&#x20;
  2. **Connection to Previous:** Contrasts mesh connectivity with topology‑free point sets, emphasizing simplicity vs completeness.
  3. **Outlook for Next:** Leads into functional, equation‑based shape definitions.

* **Section 5 (Slides 65–end): Implicit Representations & Conversions**

  1. **Summary:**

     * **Explicit vs Implicit:** sampling vs inside/outside ease.
     * **Algebraic Surfaces:** polynomial zero sets.
     * **CSG:** Boolean ops on implicit functions.
     * **Distance Functions & Level Sets:** signed distances, blending, grid storage.
     * **Conversion:** pipelines between meshes, points, and implicit fields; “no single best” verdict.&#x20;
  2. **Connection to Previous:** Completes the spectrum from discrete point/mesh to fully continuous definitions.
  3. **Outlook for Next:** Previews 3D transformations and further geometry processing.

III. **Core Points & Memory Aids**

* **Representation Categories:** Mesh, Point Cloud, Implicit. (Aid: “MPI = Mesh‑Point‑Implicit”)&#x20;
* **Polygonal Mesh:** Vertices + edges + faces, often triangulated, must satisfy manifold conditions. (Aid: “MVN = Mesh Vertices & Neighbors”)&#x20;
* **Indexed Face Set:** Vertices list + face indices; CCW order for normals. (Aid: “IFS = Indices Fix Surfaces”)&#x20;
* **Farthest Point Sampling:** Greedy max‑distance strategy for uniform coverage. (Aid: “FPS = Furthest Picks Spread”)&#x20;
* **Normal via PCA:** Smallest eigenvector of local covariance = surface normal. (Aid: “PCA → Principal Component = ‘Correct Angle’”)&#x20;
* **Implicit Surface f(x,y,z)=0:** Zero‑level set defines boundary; trivial inside/outside test. (Aid: “Imply = Implicit → ‘In/Out’ test”)&#x20;
* **Signed Distance Function:** Distance‑based blending and CSG. (Aid: “DS = Distance Shapes”)&#x20;

IV. **Pre-Study Suggestions**

1. **Mesh Practice:** Write a Python script to load/create a simple triangular mesh, compute and visualize vertex normals.
2. **Acquisition Primer:** Read a tutorial on LIDAR vs. structured‑light scanners (e.g., Kinect) to grasp point‑cloud noise and occlusion.
3. **Implicit Modeling:** Define f(x,y,z)=x²+y²+z²−1 in code; experiment with a marching‑cubes library to extract the sphere mesh.

V. **Review Questions/Practice**

1. **Comparison Task:** For a given object, compare memory/storage cost and query complexity across multiview, mesh, point cloud, and volumetric formats.
2. **Sampling Challenge:** Implement farthest‑point sampling on a mesh‑derived point set and plot the sampling outcome.
3. **CSG Exercise:** Derive and plot the signed distance field for the union of two overlapping spheres using your language of choice.

## Lecture 3

I. **Overall Overview**

* **Topic:** 3D Transformations in Visual Computing – linear, affine, and Euclidean transforms, homogeneous‑coordinate formulation, and rotation representations.&#x20;
* **Learning Objectives:**

  1. Grasp the properties of linear vs. affine transformations and basic building blocks (scale, rotation, translation, reflection, shear).
  2. Learn how homogeneous coordinates unify translation with other linear maps in 2D and 3D.
  3. Understand the structure of the rotation groups SO(2) and SO(3), and compare parameterizations: Euler angles, axis–angle (exponential map), and quaternions.
* **Core Thread:** Starting from the algebraic axioms of linearity, we see how all common transforms can be represented as matrices, extend this to handle translation via homogeneous coordinates, then dive deep into the math and topology of rotations—culminating in practical rotation parameterizations and their trade‑offs for graphics and learning systems.&#x20;

II. **Section Summaries**

* **Section 1 (Basic Transforms & Composition, Slides 1–14)**
  Linear maps satisfy additivity and homogeneity, making scale \$(S\_a(x)=ax)\$ and rotation linear, but translation is not. We introduce reflection and shear, classify transforms as linear, affine (linear+translation), Euclidean (distance‑preserving), and rigid (Euclidean + no reflection), and show how to compose them.&#x20;

* **Section 2 (2D Homogeneous Coordinates, Slides 15–24)**
  Embed 2D points as \$(x,y,1)^T\$ so that translations become 3×3 matrix multiplications. Scale and rotation extend trivially; translation becomes a shear in homogeneous space. Recover true 2D by dividing by the third component.&#x20;

* **Section 3 (3D Homogeneous Coordinates, Slides 25–34)**
  Lift 3D points to \$(x,y,z,1)^T\$ and represent all affine transforms with 4×4 matrices. Examples: scale diag\$(S\_x,S\_y,S\_z,1)\$, translation with last column \$(b\_x,b\_y,b\_z,1)\$, shear along \$x\$ using \$y,z\$ offsets, and around‐axis rotation matrices.&#x20;

* **Section 4 (Rotation Groups & SO(n), Slides 35–44)**
  Define \$\mathrm{SO}(n)={R\in\mathbb R^{n\times n}\mid R^\top R=I,\det R=1}\$. Contrast the 1‑DOF circle topology of \$\mathrm{SO}(2)\$ with the 3‑DOF non‑contractible manifold of \$\mathrm{SO}(3)\$. Highlight pitfalls when mapping from Euclidean domains into these curved groups in neural networks.&#x20;

* **Section 5 (Euler‐Angle Representation, Slides 45–52)**
  Parameterize \$R=R\_z(\gamma)R\_y(\beta)R\_x(\alpha)\$ via three sequential axis rotations. Intuitive and compact but non‑unique (gimbal lock when \$\beta=\pi/2\$) and discontinuous, making it problematic for smooth optimization.&#x20;

* **Section 6 (Axis–Angle & Exponential Map, Slides 53–62)**
  By Euler’s theorem any \$R\in\mathrm{SO}(3)\$ is \$\exp(\[\hat\omega]\theta)\$: rotate about unit axis \$\hat\omega\$ by angle \$\theta\$. Use the skew‑symmetric $\[\hat\omega]\$ and Rodrigues’ formula \$R=I+\[\hat\omega]\sin\theta+\[\hat\omega]^2(1-\cos\theta)\$. Extract \$(\hat\omega,\theta)\$ from \$R\$ except at singular cases (\$\theta=0,\pi\$), and define the natural rotation distance \$\theta(R\_1R\_2^\top)\$.&#x20;

* **Section 7 (Quaternions, Slides 63–71)**
  Quaternions \$q=w+xi+yj+zk\$ form a 4D algebra with \$i^2=j^2=k^2=ijk=-1\$. Unit quaternions (\$|q|=1\$) double‑cover \$\mathrm{SO}(3)\$ and rotate a vector \$\mathbf x\$ via \$q,(0,\mathbf x),q^{-1}\$. Convert between axis‐angle $\[\cos\frac\theta2,\sin\frac\theta2\hat\omega]\$ and rotation matrices; note the need to normalize in networks and the “double‑cover” ambiguity.&#x20;

III. **Core Points & Memory Aids**

* **Linearity**: \$f(x+y)=f(x)+f(y)\$, \$f(\alpha x)=\alpha f(x)\$.
  *(Aid: “LAH” – **L**inear = **A**dditive + **H**omogeneous)*
* **Affine = Linear + Translation**: transforms that can’t be purely linear.
  *(Aid: “**A** ffine adds **A**fter”)*
* **Homogeneous Coordinates**: embed \$(x,y)\to(x,y,1)\$ so translation is linear.
  *(Aid: “**HOMO** makes **HO** TRANSforms linear)*
* **\$\mathrm{SO}(3)\$**: special orthogonal group (det = 1, \$R^\top R=I\$).
  *(Aid: “**SO**unds **O**rthogonal”)*
* **Euler Angles**: three sequential axis rotations; intuitive but “gimbal locks.”
  *(Aid: “**E**asy to **U**nderstand, but **E**erie locks”)*
* **Axis–Angle (Exp Map)**: \$R=e^{\[\hat\omega]\theta}\$ via Rodrigues.
  *(Aid: “**AA** – **A**xis and **A**ngle = Always Attitude”)*
* **Quaternions**: 4‑vector compact rotation, \$q x q^{-1}\$.
  *(Aid: “**Q**uick **A**ttitude **T**ool”)*
* **Rodrigues’ Formula**: \$R=I+\[\hat\omega]\sin\theta+\[\hat\omega]^2(1-\cos\theta)\$.
  *(Aid: “**Rod** rides on demand”)*

IV. **Pre-Study Suggestions**

1. **Linear Algebra Refresher:** Re‑derive why translation isn’t linear; practice expressing scale, rotation, shear, reflection as small matrices.
2. **2D Homogeneous Coding:** In Python/NumPy, build 3×3 homogeneous‐coordinate transforms for scale, rotate, translate, and visualize their action on simple polygons.
3. **Skew‐Symmetric & Exponential:** Review cross product as $\[v]w=v\times w\$ and implement \$\exp(\[v]\theta)\$ via Rodrigues’ series expansion.

V. **Review Questions/Practice**

1. **Rotation‑Matrix → Axis‑Angle:** Given a 3×3 \$R\$, derive and code extraction of \$(\hat\omega,\theta)\$, handling the \$\theta\approx0,\pi\$ edge cases.
2. **Quaternion vs. Euler:** Implement both quaternion and Euler‐angle rotations on a 3D mesh; compare interpolation smoothness and numerical stability.
3. **SLERP Exercise:** Code spherical linear interpolation (SLERP) between two unit quaternions and apply it to animate a rigid body’s orientation.

## Lecture 4

I. **Overall Overview**

* **Topic:** Learning‑based Structure from Motion (SfM)—classical SfM pipeline and modern deep‑learning enhancements.&#x20;
* **Learning Objectives:**

  1. Master the projection and epipolar geometry that underlies two‑view SfM (pinhole camera model, ray back‑projection, essential matrix).&#x20;
  2. Understand the classical SfM pipeline variants—incremental, global, hierarchical—including feature extraction, matching, triangulation, bundle adjustment, and rotation averaging.&#x20;
  3. Explore how learned modules (SuperPoint for detection/description and SuperGlue for matching) can improve robustness and sub‑pixel precision in SfM.&#x20;
* **Core Thread:** The lecture first revisits 2D–3D mapping and two‑view geometry, then walks through SfM system design (pipeline, paradigms, and algorithmic details), and finally presents data‑driven approaches—SuperPoint and SuperGlue—that augment key SfM stages for greater accuracy in challenging scenarios.&#x20;

II. **Section Summaries**

* **Section 1 (Slides 1–10): Recap & 2D–3D Basics**

  1. **Summary:** Quick review of homogeneous transforms and rotation representations; pinhole projection $x = K [R\mid t] X$; back‑projection rays; ray–ray intersection for depth; and the essential matrix constraint $x'^{T}E\,x=0$.&#x20;
  2. **Connection to Previous:** Builds directly on Lecture 3’s transforms and SO(3) theory by applying them to camera geometry.
  3. **Outlook for Next:** Sets up the need to recover scene structure and camera poses from multiple images.

* **Section 2 (Slides 11–15): SfM Pipeline & Paradigms**

  1. **Summary:** Presents the overall SfM workflow—unstructured images → scene graph → sparse model → dense model → multi‑view stereo (MVS)—and contrasts the three paradigms: incremental, global, and hierarchical SfM.&#x20;
  2. **Connection to Previous:** Translates two‑view geometry into a scalable multi‑image reconstruction framework.
  3. **Outlook for Next:** Delves into the steps of incremental SfM in detail.

* **Section 3 (Slides 16–30): Incremental SfM Details**

  1. **Summary:** Step‑by‑step pipeline: feature extraction → matching → geometric verification → image registration → initialization (pick two views, triangulate inliers, bundle adjustment) → incremental addition of new views via PnP → global bundle adjustment → outlier filtering.&#x20;
  2. **Connection to Previous:** Implements the pipeline sketched before, showing how relative two‑view poses bootstrap full scene reconstruction.
  3. **Outlook for Next:** After incremental methods, we’ll explore global rotation and translation estimation.

* **Section 4 (Slides 31–36): Global SfM & SfM vs SLAM**

  1. **Summary:** Global SfM solves all camera rotations via rotation averaging (minimizing $\sum\|\!R_iR_j^T\!−R_{ij}\|$) then recovers translations; compares SfM (offline, high‑precision) with SLAM (online, real‑time).&#x20;
  2. **Connection to Previous:** Offers an alternative to incremental chaining, providing robustness to drift via global optimization.
  3. **Outlook for Next:** Motivates hybrid pipelines and the introduction of learned modules to enhance each stage.

* **Section 5 (Slides 37–45): SuperPoint—Learned Keypoints & Descriptors**

  1. **Summary:** SuperPoint is a single ConvNet that jointly detects interest points (via an 8×8 grid classifier + NMS) and computes descriptors by bilinear interpolation; trained first on synthetic “MagicPoint” data then self‑supervised on COCO via homographic adaptation.&#x20;
  2. **Connection to Previous:** Replaces classical detectors (SIFT, ORB) to yield more repeatable and distinctive keypoints under challenging conditions.
  3. **Outlook for Next:** Shows how to leverage context for matching with SuperGlue.

* **Section 6 (Slides 46–56): SuperGlue—Contextual Feature Matching**

  1. **Summary:** SuperGlue formulates matching as a graph‑based partial assignment: features (coords + descriptors) pass through self‑ and cross‑attention layers, then a differentiable Sinkhorn solver yields soft one‑to‑one correspondences with an extra “dustbin” for outliers.&#x20;
  2. **Connection to Previous:** Builds on SuperPoint’s outputs, introducing global reasoning to filter out mismatches and handle occlusions.
  3. **Outlook for Next:** Encourages thinking about end‑to‑end learned SfM and future modules for triangulation and bundle adjustment.

* **Section 7 (Slides 57–end): Conclusions & Next Steps**

  1. **Summary:** Reflects on the integration of learned modules within classical SfM, surveys remaining challenges (few‑view settings, end‑to‑end learning), and previews next lecture on dense multi‑view stereo.&#x20;
  2. **Connection to Previous:** Links back to the lecture’s opening outline and suggests the direction toward fully‑learned 3D reconstruction.
  3. **Outlook for Next:** Dense modeling and neural rendering topics.

III. **Core Points & Memory Aids**

* **Pinhole Projection $x = K[R\!\mid\!t]X$**: maps 3D to 2D via intrinsics + extrinsics.
  *(Aid: “KRT = Keep Riding Transforms”) *
* **Essential Matrix $x'^{T}E\,x=0$**: encodes epipolar constraint.
  *(Aid: “E=Epipolar line Enforcer”) *
* **Incremental SfM Steps**: Detect → Match → Verify → Triangulate → BA.
  *(Aid: “DMTVB = Don’t Miss The Very Bundle”) *
* **Bundle Adjustment**: joint nonlinear refine of all poses & points to minimize reprojection error.
  *(Aid: “BA = Best Alignment”) *
* **Rotation Averaging**: solve for all $R_i$ by minimizing $\sum\|R_iR_j^T−R_{ij}\|$.
  *(Aid: “RA = Rotations Aligned”) *
* **SuperPoint**: grid‑based keypoint detector + descriptor head under one ConvNet.
  *(Aid: “SP = Single Point”) *
* **SuperGlue**: graph GNN + Sinkhorn for soft assignment of keypoints.
  *(Aid: “SG = Soft Glue”) *

IV. **Pre-Study Suggestions**

1. **Camera Model Derivation:** Derive the pinhole projection and back‑projection ray equations; implement in Python.
2. **Essential Matrix Estimation:** Review the eight‑point algorithm and its SVD solution for $E$.
3. **Optimization Primer:** Refresh Gauss–Newton and Levenberg–Marquardt methods used in bundle adjustment.

V. **Review Questions/Practice**

1. **Build a Mini SfM:** Using OpenCV, detect ORB features in two images, estimate $E$, recover pose, triangulate points, and display a sparse 3D cloud.
2. **Rotation Averaging:** Given noisy relative rotations among 5 cameras, implement a simple least‑squares rotation averaging and evaluate consistency.
3. **Train SuperPoint on Synthetic Data:** Generate random homographies on a checkerboard, train a toy keypoint detector, and compare repeatability to Harris corners.

## Lecture 5

I. **Overall Overview**

* **Topic:** Learning‑Based Multi‑View Stereo (MVS)
* **Learning Objectives:**

  1. Understand classical image‑based 3D reconstruction methods (silhouette carving, photometric consistency, plane‑sweep, fusion).
  2. Learn how deep networks (MVSNet) build and process cost volumes for dense depth estimation, and how to enhance them via adaptive sampling, point‑based refinement, and depth‑normal consistency.
* **Core Thread:** Building on SfM’s sparse point recovery, this lecture surveys traditional MVS techniques for dense shape reconstruction, introduces volumetric deep‑learning pipelines (cost‑volume + 3D CNN), and then presents key improvements—adaptive thin‑volumes, point‑flow refinement, and depth‑normal regularization—to boost accuracy and efficiency.&#x20;

II. **Section Summaries**

* **Section 1 (Slides 1–5): Recap & Lecture Outline**

  1. **Summary:** Title slide, semester roadmap, brief recap of SfM fundamentals (pinhole projection, essential matrix, incremental/global SfM), and transition to dense 3D reconstruction via MVS.&#x20;
  2. **Connection to Previous:** Bridges from sparse camera‑pose and point‑cloud recovery (Lecture 4) to the need for per‑pixel depth.
  3. **Outlook for Next:** Survey of classical dense reconstruction techniques.

* **Section 2 (Slides 6–17): Classical MVS Techniques**

  1. **Summary:**

     * **Silhouette Carving:** back‑project object silhouettes to build visual hulls.
     * **Photometric Consistency:** match intensities across views and triangulate (plane‑sweep stereo).
     * **Depth‑Map Merging & Volumetric Fusion:** combine per‑view depth into a unified mesh (Curless–Levoy).
     * **Multi‑ vs Two‑View:** more views reduce error/occlusions; view‑selection for unstructured collections.&#x20;
  2. **Connection to Previous:** Applies epipolar and triangulation theory from SfM to dense per‑pixel matching.
  3. **Outlook for Next:** Highlights failure modes in low‑texture or reflective areas, motivating learned features.

* **Section 3 (Slides 18–22): Volumetric Deep MVS (MVSNet)**

  1. **Summary:**

     * Build a **cost volume** by warping learned 2D features into a reference frustum.
     * Apply a **3D CNN** to regularize along depth hypotheses.
     * Predict depth via differentiable soft‑argmax (weighted sum), yielding per‑pixel depth maps.
     * **Pipeline trade‑offs:** robust matching vs. high compute/memory and “flying points.”&#x20;
  2. **Connection to Previous:** Transforms classical photometric cost into a learned volumetric aggregation.
  3. **Outlook for Next:** Reducing volume size and speeding up inference via adaptive sampling.

* **Section 4 (Slides 23–28): Adaptive Sampling & Point‑Based Refinement**

  1. **Summary:**

     * **Adaptive Thin‑Volume:** use per‑pixel uncertainty to focus depth hypotheses (coarse‑to‑fine).
     * **Cascaded Prediction:** multi‑scale networks that iteratively refine depth ranges.
     * **PointFlow MVS:** unproject coarse depth into sparse point cloud and learn per‑point offsets along rays.&#x20;
  2. **Connection to Previous:** Addresses MVSNet’s inefficiency by narrowing search and moving to sparse representations.
  3. **Outlook for Next:** Tackles surface smoothness by incorporating normals.

* **Section 5 (Slides 29–36): Depth‑Normal Consistency**

  1. **Summary:**

     * Depth supervision alone yields noisy, non‑smooth surfaces.
     * Predict **surface normals** as an auxiliary task—normals easier to learn.
     * Enforce **depth‑normal consistency** via a local plane constraint and use normals to correct depth estimates.&#x20;
  2. **Connection to Previous:** Merges geometric priors (Lecture 1 curvature concepts) with deep depth estimation.
  3. **Outlook for Next:** Prepares for dense fusion and neural rendering in upcoming lectures.

III. **Core Points & Memory Aids**

* **Silhouette Carving:** back‑project silhouettes ∩ hull (Aid: “Silhouette = Shape Slice”)&#x20;
* **Photometric Consistency:** match intensities across views (Aid: “Photo = Pixel Pact”)&#x20;
* **Plane‑Sweep Stereo:** sweep depth planes, compute cost volume (Aid: “Sweep = Surface Scan”)&#x20;
* **MVSNet Cost Volume:** volumetric feature aggregation via 3D CNN (Aid: “CV = Compute Volume”)&#x20;
* **Adaptive Sampling:** thin‑volume coarse‑to‑fine (Aid: “Thin = Think Thin”)&#x20;
* **PointFlow:** refine depth via point offsets (Aid: “Flow = Fine Locally Optimizing World”)&#x20;
* **Depth‑Normal Consistency:** enforce local planar fit (Aid: “DN = Depth & Normals Dance”)&#x20;

IV. **Pre-Study Suggestions**

1. Review **Curless–Levoy** volumetric fusion and implement a simple silhouette intersection.
2. Read the original **MVSNet** paper to understand feature warping and soft‑argmax depth regression.
3. Brush up on **3D CNNs** and point‑cloud networks (e.g., PointNet).

V. **Review Questions/Practice**

1. Compare classical photometric stereo vs. MVSNet depth maps on a textured object: analyze error and runtime.
2. Implement a coarse‑to‑fine depth sampling module: narrow a uniform depth range based on prior confidence.
3. Design and test a consistency loss combining predicted depth and normals on synthetic data to evaluate smoothness gains.

## Lecture 6

I. **Overall Overview**

* **Topic:** Neural Radiance Fields (NeRF) and Beyond—implicit scene representations and differentiable volume rendering for photo‑realistic view synthesis.
* **Learning Objectives:**

  1. Understand implicit function‑based 3D shape representations (signed distance fields, mixture models, learned MLPs) and their pros/cons.
  2. Master the volumetric light transport model and its discretization via ray marching, then learn how NeRF repurposes an MLP to predict density and radiance for each 3D sample.
  3. Explore key training “tricks” (hierarchical sampling, Fourier feature positional encoding) and survey extensions to dynamic scenes (D‑NeRF) and few‑shot generalization (PixelNeRF).
* **Core Thread:** After a quick recap of MVS and rendering basics, we first ground ourselves in implicit geometry representations; next, we derive the volume rendering integral and its ray‑marching discretization; then we see how NeRF uses an MLP to learn a continuous radiance field via differentiable rendering; finally, we examine extensions that add time or learn priors for one‑shot synthesis.&#x20;

II. **Section Summaries**

**Section 1 (Slides 1–6): Intro, Recap & Rendering Basics**

1. **Summary:** Announces Homework 2 and reviews learned MVS techniques (classic/learning‑based, adaptive sampling, depth‑normal consistency). Introduces “Today’s Focus” on photo‑realistic rendering—scene/materials/lighting, camera extrinsics/intrinsics—and the rendering equation.&#x20;
2. **Connection to Previous:** Transitions from sparse MVS reconstructions to dense image synthesis, building on camera models and geometry from Lectures 3–5.
3. **Outlook for Next:** Motivates implicit scene encoding as a continuous alternative to classical volumetric representations.

**Section 2 (Slides 7–18): Implicit 3D Shape Representations**

1. **Summary:** Contrasts explicit (meshes, point clouds, voxels) with implicit definitions (zero‑level sets of F(x)), introduces signed distance fields (SDFs) and mixture‑of‑Gaussian fields, then surveys learned MLP‑based implicits (DeepSDF, Occupancy Nets, PIFu). Pros (continuous, analytic normals, topology‑agnostic) and cons (watertight requirement, extra visualization steps).&#x20;
2. **Connection to Previous:** Extends classical mesh/point‐cloud concepts (Lecture 2) into continuous functional form, preparing for neural volume rendering.
3. **Outlook for Next:** Leads into how these implicit densities feed into a volumetric light transport model.

**Section 3 (Slides 19–29): Volume Rendering & Ray Marching**

1. **Summary:** Derives the volumetric radiance integral along a camera ray—accounting for transmission (Beer–Lambert) and emission—then discretizes via ray marching:

   $$
   I = \sum_{i=1}^n T_i\,\alpha_i\,\frac{c_i}{\sigma_i},\quad T_i=\prod_{j>i}(1-\alpha_j).
   $$
2. **Connection to Previous:** Applies implicit density fields to image formation, bridging geometry and photometry.
3. **Outlook for Next:** Shows how to replace handcrafted densities with a learned MLP mapping in NeRF.&#x20;

**Section 4 (Slides 30–38): Learning NeRF**

1. **Summary:** Defines NeRF as an MLP $F_\Theta(x,y,z,\theta,\phi)\to(\sigma,\;c/\sigma)$, uses differentiable ray marching and a per‑pixel loss to optimize network weights for one scene. Covers training pipeline, hierarchical (coarse‑to‑fine) sampling for efficiency, and Fourier‑feature positional encoding to capture high‑frequency details. Concludes with “NeRF in a Nutshell” summary and shows exemplar results, noting challenges (dynamic scenes, per‑scene training).&#x20;
2. **Connection to Previous:** Replaces classical cost volumes (Lecture 5) with a continuous neural field, leveraging the volume rendering framework just derived.
3. **Outlook for Next:** Surveys extensions that enable dynamics or cross‑scene generalization.

**Section 5 (Slides 39–end): NeRF Extensions & Future Directions**

1. **Summary:**

   * **D‑NeRF:** Adds time as a 6th input dimension plus a deformation network to handle non‑rigid dynamics in a canonical space.
   * **PixelNeRF:** Trains across many scenes to learn a prior, enabling few‑shot view synthesis from sparse inputs.
   * **AIGC & 3DGS:** Sketches DREAMFUSION for text‑to‑3D and 3D Gaussian Splatting for fast rendering via rasterized Gaussian primitives.
2. **Connection to Previous:** Builds directly on the NeRF architecture, showing how to inject temporal or cross‑scene priors and alternative rendering paradigms.
3. **Outlook for Next:** Points toward neural rendering for animation, real‑time novel‑view synthesis, and integration with generative models.&#x20;

III. **Core Points & Memory Aids**

* **Signed Distance Field (SDF):** $F(p)\!=\!\pm\min\|p-s\|$ with $F=0$ at surface.
  *(Aid: “SDF—**S**ign **D**efines **F**rontier”)*&#x20;
* **Volume Rendering Integral:** $I=\int T(t)\,\sigma(t)\,c(t)\,\mathrm dt$.
  *(Aid: “**VRI**—**V**olume **R**adiance **I**ntegral”)*&#x20;
* **Ray Marching Discretization:** $I=\sum_i T_i\,\alpha_i\,(c_i/\sigma_i)$.
  *(Aid: “**RM**—**R**ay **M**akes **S**um”)*&#x20;
* **NeRF MLP Mapping:** $(\sigma,c/\sigma)=F_\Theta(x,y,z,\theta,\phi)$.
  *(Aid: “**Ne**ural **R**adiance **F**ield”)*&#x20;
* **Hierarchical Sampling:** coarse‑to‑fine importance sampling of depths.
  *(Aid: “**HS**—**H**unt **S**harpness”)*&#x20;
* **Positional Encoding:** Fourier‑feature mapping to capture high frequencies.
  *(Aid: “**PE**—**P**eriodic **E**xpansion”)*&#x20;
* **D‑NeRF:** 6D radiance field with deformation network.
  *(Aid: “**D**ynamics‑NeRF”)*&#x20;
* **PixelNeRF:** Cross‑scene NeRF for few‑shot synthesis.
  *(Aid: “**P**rior‑NeRF”)*&#x20;

IV. **Pre-Study Suggestions**

1. **Volume Rendering Math:** Re‑derive the continuous light transport integral and its discretization under Beer–Lambert law.
2. **NeRF Paper:** Read Mildenhall *et al.* “NeRF: Representing Scenes as Neural Radiance Fields” to understand architecture and training details.
3. **Ray Marching Prototype:** Implement a simple Python ray marcher over a synthetic density field (e.g., 3D Gaussian) to visualize accumulated radiance.

V. **Review Questions/Practice**

1. **Derivation Exercise:** From first principles, derive the general form $I=\sum_iT_i\alpha_i(c_i/\sigma_i)$ and implement it in code.
2. **Mini‑NeRF Training:** Use an existing NeRF codebase to train on a small, multi‑view toy scene; evaluate the effect of hierarchical vs. uniform sampling.
3. **Extension Design:** Propose and sketch how you would incorporate surface reflectance (BRDF) into the NeRF pipeline for relighting capability.

## Lecture 7

I. **Overall Overview**

* **Topic:** Single‑Image to 3D Reconstruction — learning‑based pipelines for inferring 3D shape (point clouds and meshes) from a single RGB image.
* **Learning Objectives:**

  1. Identify the monocular cues and data‐synthesis strategies enabling single‑image 3D learning, and understand the real vs. synthetic data trade‑offs.
  2. Learn deep network approaches for point‑cloud generation (set prediction with EMD/CD losses) and mesh reconstruction (template deformation with regularization losses).
* **Core Thread:** This lecture frames monocular 3D as a severely under‑constrained inverse problem, shows how large‑scale synthetic data (ShapeNet, Objaverse) and rendering pipelines supply training pairs, then dives into deep models: first predicting unordered point sets—necessitating permutation‑invariant losses (EMD, Chamfer) and two‑branch architectures—and then extending to mesh outputs via editing‑based deformation with specialized smoothness and normal consistency losses.&#x20;

II. **Section 1 (Slides 1–12)**

1. **Summary:**

   * Recaps neural rendering (NeRF) and implicit representations.
   * Poses the core task: infer 3D from one image using monocular cues (contrast, color, texture, motion, symmetry, semantic priors).
   * Introduces the **synthesis‑for‑learning** pipeline: real sensors (Kinect, LiDAR) vs. large synthetic shape datasets (ShapeNet, Objaverse) rendered into 2D.<br>
2. **Connection to Previous:** Leverages volume‑rendering and implicit‐function tools from NeRF (Lectures 6) to motivate monocular tasks and data needs.
3. **Outlook for Next:** Transition to concrete single‑image reconstruction methods, starting with point‑cloud prediction.&#x20;

III. **Section 2 (Slides 13–53)**

1. **Summary:**

   * **Point‑Cloud Generation (PSGN):** CNN → N×3 point set; requires set losses.
   * **Permutation Invariance:** point clouds have no ordering → need metrics like **Earth Mover’s Distance** (EMD) and **Chamfer Distance** (CD); trade‑offs in continuity, differentiability, and speed.
   * **Network Design:** two‑branch architectures combining up‑convolutions for smooth regions and fully‑connected branches for detailed structures, effectively learning a surface parameterization.
2. **Connection to Previous:** Extends point‐cloud fundamentals (Lecture 2) and set‐metric ideas to monocular prediction, applying neural‐rendering‑style supervision.
3. **Outlook for Next:** Having generated geometry as points, we now tackle full **mesh** reconstruction with connectivity.&#x20;

IV. **Section 3 (Slides 54–end)**

1. **Summary:**

   * **Mesh Challenges:** regressing vertices **and** edges is ambiguous due to topology variations.
   * **Editing‑Based Modeling:** deform a template mesh via learned per‑vertex offsets Δp, sidestepping edge prediction.
   * **Mesh Regularization Losses:**

     * **Set Distance:** EMD/CD on vertices
     * **Uniformity:** edge‑length regularizer for even sampling
     * **Smoothness:** penalize dihedral angles deviating from 180°
     * **Normal Consistency:** enforce local tangent‑plane alignment using ground‑truth normals
2. **Connection to Previous:** Builds on point‑set generation and depth‑normal consistency (Lecture 5), adding topology and higher‑order surface priors.
3. **Outlook for Next:** With template deformation covered, future lectures will explore fully implicit mesh recovery and dynamic/topology‑aware models.&#x20;

V. **Core Points & Memory Aids**

* **Monocular Cues:** contrast, texture, symmetry, semantic priors (Aid: “CTS SP – Cats Sleep Precisely”)&#x20;
* **Synthesis‑for‑Learning Pipeline:** real sensors vs. synthetic renders (Aid: “Real or Render”)&#x20;
* **Permutation Invariance:** unordered points need set metrics (Aid: “PI – Points Indistinct”)&#x20;
* **Earth Mover’s Distance (EMD):** minimal transport cost (Aid: “EMD – Earth Moves Dirt”)&#x20;
* **Chamfer Distance (CD):** nearest‑neighbor sum (Aid: “CD – Closest Distance”)&#x20;
* **Two‑Branch Architecture:** upconv + FC for smooth vs. detailed geometry (Aid: “UB – Union Branch”)&#x20;
* **Editing‑Based Mesh Modeling:** deform template via Δp (Aid: “EMM – Edit Mesh Manually”)&#x20;
* **Mesh Regularization:** set, uniformity, smoothness, normal losses (Aid: “USSN – Uniform Smooth Normals”)&#x20;

VI. **Pre-Study Suggestions**

1. **Monocular Depth Review:** revisit classical depth‐normal consistency from GeoNet (Qi et al. CVPR 2018).&#x20;
2. **PSGN Paper:** read “A Point Set Generation Network for 3D Object Reconstruction from a Single Image” (Fan et al., CVPR 2017).&#x20;
3. **Mesh Fundamentals:** brush up on mesh data structures (vertices, edges, faces), manifold vs non‑manifold, and basic deformation tools.&#x20;

VII. **Review Questions/Practice**

1. **Implement Set Losses:** code both Chamfer and EMD on toy point sets; visualize gradient behavior.&#x20;
2. **Train a Mini‑PSGN:** on simple shapes (cube, sphere) with CD vs EMD; compare reconstructed biases.&#x20;
3. **Template Deformation:** starting from a unit‑sphere mesh, apply learned Δp with set/uniformity/smoothness losses to morph into a cylinder; evaluate quality.&#x20;

## Lecture 8

I. **Overall Overview**

* **Topic:** Surface Completion—reconstructing triangle meshes from raw point clouds.
* **Learning Objectives:**

  1. Understand the surface‑completion problem setup, desired mesh properties (manifold, watertight), and performance criteria (speed, robustness).
  2. Learn two families of mesh‑reconstruction methods: explicit algorithms (rule‑based and learning‑enhanced) and implicit field‑based algorithms (classical and neural).
* **Core Thread:** This lecture frames surface completion as converting unstructured point clouds into high‑quality meshes under geometric constraints. It first explores explicit rule‑based (ball‑pivoting) and learned connectivity methods for triangle formation, then delves into implicit field approaches—RBF, MLS, Poisson—and finally covers neural implicit techniques (DeepSDF, SAL) for robust, watertight reconstructions.&#x20;

II. **Section 1 (Slides 1–8): Problem Definition & Constraints**

1. **Summary:** Defines input (point cloud ± normals) and desired output (manifold, watertight triangle mesh). Reviews desirable properties: fast runtime, robustness to noise, plus geometric constraints—no self‑intersections, non‑manifold edges/vertices, and watertightness.&#x20;
2. **Connection to Previous:** Builds on Lecture 7’s mesh‑editing and Lecture 2’s point‑cloud fundamentals by formalizing the reverse task of mesh completion. Prerequisite: familiarity with point clouds, mesh topology, and normal estimation.
3. **Outlook for Next:** Leads into explicit reconstruction methods that directly connect points into triangles.

III. **Section 2 (Slides 9–25): Explicit Algorithms**

1. **Summary:**

   * **Ball‑Pivoting Algorithm:** Grow triangles by rolling a ball of radius ρ over the point set—three touched points form a face; iterate with multiple ρ values to capture fine details and close holes.
   * **Limitations & Learning‑Based Enhancement:** Rule‑based methods fail on ambiguous thin structures; a network trained with an intrinsic‑extrinsic ratio guides correct triangle connections. Pros: linear time, no normals; Cons: potential non‑manifold meshes, no watertight guarantee, sensitive to sampling density.&#x20;
2. **Connection to Previous:** Applies the geometric constraints from Section 1 to concrete triangle‑formation rules, illustrating how learned filters can overcome purely heuristic failures.
3. **Outlook for Next:** Transitions to implicit field methods that avoid explicit connectivity rules.

IV. **Section 3 (Slides 26–end): Implicit Algorithms**

1. **Summary:**

   * **Classical Implicit:** Define surface as zero‑level set $F(x)=0$. Reconstruct via:

     * **Radial Basis Functions (RBF):** global interpolation $f(x)=\sum_i\omega_i\phi(\|x-x_i\|)+p(x)$ with off‑surface constraints from normals.
     * **Moving Least Squares (MLS):** local weighted least squares for smooth implicit surfaces.
     * **Poisson Surface Reconstruction:** solve $\nabla^2 f = \nabla\cdot n$ for robust, watertight meshes; screened variant adds sharpness.
     * Extract iso‑surfaces via Marching Cubes (topology‑corrected variants).
   * **Neural Implicit:**

     * **DeepSDF:** overfit or learn latent‑conditioned signed distance fields with an MLP.
     * **SAL (Sign‑Agnostic Learning):** learn signed fields from unsigned data using a custom loss to avoid trivial minima.
2. **Connection to Previous:** Contrasts explicit connectivity with continuous field estimation, leveraging implicit‑function basics from Lectures 1 & 6 and mesh‑extraction tools (marching cubes). Prerequisite: SDF concepts, marching cubes, MLP familiarity.
3. **Outlook for Next:** Points toward generative 3D (AIGC) and dynamic implicit fields in future lectures.

V. **Core Points & Memory Aids**

* **Manifold Mesh:** no self‑intersection, each edge exactly two faces, connected one‑ring.
  *(Aid: “**Mani‑fold**: only one fold allowed”)*
* **Watertightness:** every edge has two incident faces → defines a solid’s interior.
  *(Aid: “**Water**‑tight: no leaks”)*
* **Ball‑Pivoting Algorithm:** roll a ρ‑radius ball to form triangles from three contact points.
  *(Aid: “**Ball** pivots to build faces”)*
* **Intrinsic‑Extrinsic Ratio Guidance:** learned filter for correct edge connectivity in ambiguous regions.
  *(Aid: “**I‑E Ratio** decides In vs. External links”)*
* **Implicit Zero‑Level Set:** surface = $\{x\mid F(x)=0\}$.
  *(Aid: “**Zero** marks the surface”)*
* **RBF Reconstruction:** global basis functions $f(x)=\sum\omega_i\phi(\|x-x_i\|)+p(x)$ with off‑surface normals constraints.
  *(Aid: “**RBF**: Rings Build Functions”)*
* **Moving Least Squares (MLS):** local, weight‑driven least squares for smooth F(x).
  *(Aid: “**MLS**: Move Local Smoothly”)*
* **DeepSDF & SAL:** neural MLPs to learn signed/unsigned distance fields, enabling watertight iso‑surface extraction.
  *(Aid: “**SAL** un‑Signs the data”)*

VI. **Pre-Study Suggestions**

1. **Mesh Topology Refresher:** Review definitions of manifold vs. non‑manifold meshes and watertightness.&#x20;
2. **Ball‑Pivoting Read:** Skim “Ball‑Pivoting for Surface Reconstruction” (Bernardini *et al.*) to understand its geometric assumptions.&#x20;
3. **Implicit Surface Basics:** Refresh signed distance fields, RBF interpolation, and Poisson reconstruction fundamentals.&#x20;

VII. **Review Questions/Practice**

1. **Implement Ball‑Pivoting:** On a synthetic sphere point cloud, vary ρ and analyze mesh completeness vs. holes.&#x20;
2. **Compare Implicit Methods:** Reconstruct a noisy scan with RBF, MLS, and Poisson; evaluate error, smoothness, and runtime.&#x20;
3. **Train a Mini‑DeepSDF:** Overfit an MLP to a simple CAD shape; extract mesh via marching cubes and assess watertightness.&#x20;

## Lecture 9

I. **Overall Overview**

* **Topic:** 3D Generative Models: from classical surface reconstruction to modern neural‐implicit and deep‐learning‐based approaches for synthesizing 3D shapes from scratch and from 2D data.
* **Learning Objectives:**

  1. Review explicit vs. implicit surface reconstruction methods (e.g. ball‐pivoting, RBF/MLS, Poisson, Marching Cubes).
  2. Understand neural implicit field fitting (DeepSDF, Sign‐Agnostic Learning).
  3. Survey unconditional 3D generative paradigms (GANs, autoregressive, diffusion) and unsupervised 2D‐to‐3D methods.
* **Core Thread:**
  Beginning with classical implicit‐surface reconstruction, the lecture shows how neural networks learn continuous signed‐distance fields, then pivots to deep generative models (GANs, autoregressive, diffusion) for 3D shape synthesis—from fully supervised 3D training to purely 2D‐based unsupervised learning.

---

II. **Section Summaries**

### Section 1 (Slides 1–10): Classical Implicit vs. Explicit Meshing

1. **Summary:**

   * **Explicit (e.g. ball‐pivoting):** directly builds mesh by rolling a ball over points.
   * **Implicit:** fit a continuous field (RBF, MLS) to data, then extract zero‐isosurface via Marching Cubes.
2. **Connection to Previous:** builds on earlier lectures on point‐cloud processing and meshing.
3. **Outlook for Next:** introduces limitations (normals sensitivity, ambiguities) motivating learned implicit fields.

### Section 2 (Slides 11–22): Neural Implicit Field Fitting

1. **Summary:**

   * **DeepSDF (Park et al. CVPR’19):** learns an MLP‐parametrized signed‐distance function per shape or via latent code.
   * **SAL (Atzmon et al. CVPR’20):** sign‐agnostic learning from unsigned distances with special loss to avoid trivial minima.
2. **Connection to Previous:** extends RBF/MLS by replacing basis sums with MLPs for flexibility.
3. **Outlook for Next:** sets up using these learned fields for generation rather than just reconstruction.

### Section 3 (Slides 23–30): Unconditional 3D Generation—3D GANs

1. **Summary:**

   * **3D‐GAN (Wu et al. 2016):** voxel‐based generator & discriminator trained adversarially on ShapeNet.
   * **Results:** high‐resolution 64³ volumes, vector arithmetic in latent space.
   * **Issues:** mode collapse, vanishing gradients, training instability.
2. **Connection to Previous:** leverages DeepSDF’s continuous fields but uses discrete voxels for GAN training.
3. **Outlook for Next:** motivates alternative generative paradigms (autoregressive, diffusion) to address GAN drawbacks.

### Section 4 (Slides 31–40): Autoregressive 3D Models

1. **Summary:**

   * **AutoSDF (Mittal et al.):** transformer‐based autoregressive model over SDF codebooks for completion & generation.
   * **PolyGen (Nash et al.):** autoregressive mesh generation using sequence models.
2. **Connection to Previous:** from adversarial voxel generation to likelihood‐based sequential modeling for diversity.
3. **Outlook for Next:** highlights diffusion models as another promising likelihood‐based approach.

### Section 5 (Slides 41–52): 3D Diffusion Models

1. **Summary:**

   * **Background:** diffusion denoising process in time‐conditional U‐Nets.
   * **Point Cloud Diffusion (Luo et al.):** apply DDPMs to point cloud generation.
   * **Diffusion‐SDF:** diffuse SDF latent codes for shape completion & unconditional generation using an implicit SDF parameterization.
2. **Connection to Previous:** generalizes autoregressive likelihood to continuous diffusion sampling for smoother results.
3. **Outlook for Next:** explores 3D generation without explicit 3D supervision.

### Section 6 (Slides 53–64): Unsupervised 2D→3D Generation

1. **Summary:**

   * **HoloGAN:** learns 3D features and differentiable renderer end‐to‐end from 2D images without 3D labels.
   * **π‐GAN & EG3D:** NeRF‐based adversarial generative models with sinusoidal MLPs or hybrid representations for high‐fidelity 3D‐aware image synthesis.
2. **Connection to Previous:** removes the need for 3D training data, relying solely on 2D adversarial objectives plus 3D inductive biases.
3. **Outlook for Next:** leads naturally to text‐conditioned 3D generation via differentiable rendering and 2D diffusion guidance.

### Section 7 (Slides 65–72): Part‐Based 3D Generation

1. **Summary:**

   * **(Likely) PartNet, G2L/GRASS methods:** assemble shapes from learned part primitives to improve compositionality and diversity.
2. **Connection to Previous:** builds on unsupervised generative frameworks by adding structure via part decompositions.
3. **Outlook:** points toward interactive and conditional shape editing pipelines.

---

III. **Knowledge Connections**

* **Chunks 1→2:** Classical implicit fitting (RBF, MLS) to neural implicit fields (DeepSDF, SAL).
* **2→3:** From fitting individual shapes to sampling new shapes via adversarial training (3D‐GAN).
* **3→4:** From adversarial voxel generation to likelihood‐based autoregressive and diffusion methods for improved diversity and stability.
* **4→5:** From requiring 3D data to fully unsupervised 2D‐based generation, leveraging differentiable rendering and strong 3D priors.
* **Prerequisites:** basic calculus (iso‐surfaces), machine learning fundamentals, familiarity with MLPs, GANs, transformers, diffusion.
* **Future expansions:** text‐to‐3D (DreamFusion), real‐time 3D asset pipelines.

---

IV. **Core Points & Memory Aids**

1. **Implicit vs. Explicit Meshing:** explicit = roll‐ball (ball‐pivoting), implicit = fit field + iso‐surface (Aid: “Roll vs. Fill”).
2. **DeepSDF:** MLP learns continuous signed‐distance per shape or latent code (Aid: “Deep signed”).
3. **Sign‐Agnostic Learning (SAL):** unsigned distances → signed via special loss (Aid: “No Sign? Learn a Sign”).
4. **3D‐GAN:** voxel adversarial training for shape sampling (Aid: “3D adversary”).
5. **Autoregressive 3D:** sequentially model SDF codes or mesh vertices (Aid: “Step‐by‐Step shape”).
6. **Diffusion‐SDF:** denoise latent SDF codes over timesteps (Aid: “Diffuse the Field”).
7. **HoloGAN/π‐GAN:** learn 3D features + differentiable renderer from 2D only (Aid: “Holo 2D→3D”).
8. **DreamFusion:** optimize NeRF via 2D diffusion guidance for text‐to‐3D (Aid: “Dream of 3D”).

---

V. **Pre‑Study Suggestions**

1. **Brush up on implicit surfaces:** revisit signed‐distance functions and Marching Cubes.
2. **Review generative models:** GAN objectives, autoregressive likelihoods, diffusion fundamentals.
3. **Study differentiable rendering:** how NeRF volumetric rendering enables gradients into 3D.

---

VI. **Review Questions/Practice**

1. **Implement a simple RBF‐based implicit mesher** and extract a surface via Marching Cubes.
2. **Train a mini 3D‐GAN** on a small ShapeNet subset (e.g., chairs) and explore latent interpolations.
3. **Build a diffusion‐SDF prototype:** learn to denoise a toy 3D field (e.g., simple shapes) and visualize samples.

---

## Lecture 10

I. **Overall Overview**

* **Topic:** 3D Generative Models and 3D Backbone Networks
* **Learning Objectives:**

  1. Understand how modern 3D generative models extend GANs, autoregressive, and diffusion paradigms to 3D shape synthesis, including “unsupervised” methods that learn 3D from only 2D images.
  2. Grasp the inductive biases and architectural choices of recent 3D “backbone” networks—voxels, implicit functions (NeRF/SIREN), hybrid 3D‑2D pipelines—for tasks from generation to reconstruction.
* **Core Thread:** This lecture surveys state‑of‑the‑art generative architectures that produce or manipulate 3D shape representations, and then examines the backbone network motifs—volumetric convolutions, neural radiance fields, hybrid geometry–convolution networks—underpinning both generative and discriminative 3D vision systems.&#x20;

II. **Section Summaries**

**Section 1 (Slides 1–3): Introduction & 3D Generative Models**

1. **Summary:** Presents the landscape of 3D generative methods: 3D‑GANs, autoregressive, diffusion, unsupervised 3D from 2D, and part‑based. Emphasizes the key challenge of learning 3D shape without explicit 3D supervision.
2. **Connection to Previous:** Builds on prior lectures on 2D GANs and diffusion models by lifting them to 3D shape domains.
3. **Outlook for Next:** Transitions to the concrete architectural choices—how to represent a “3D backbone” inside these generative pipelines.

**Section 2 (Slides 4–9): Generating 3D Shapes from 2D Adversarial Nets**

1. **Summary:** Introduces HoloGAN’s architecture: start with a learned 4D constant tensor, apply 3D convolutions, explicit 3D rigid‑body transforms, a learned differentiable “projection unit,” then 2D convolutions to produce images. Discriminator sees only 2D renderings; training is unsupervised in 3D.
2. **Connection to Previous:** Extends 2D adversarial nets by folding in a volumetric 3D intermediate; reuses AdaIN and MLP style‑mapping from StyleGAN but in 3D feature space.
3. **Outlook for Next:** Motivates alternative implicit backbones—continuous fields vs. discrete voxels.

**Section 3 (Slides 10–18): Implicit Radiance‑Field Backbones (π‑GAN, EG3D…)**

1. **Summary:** Covers π‑GAN’s SIREN‑based NeRF generator: a FiLM‑conditioned MLP that outputs density σ(x) and view‑dependent color c(x,d), rendered via volume rendering; progressive growing discriminator. EG3D replaces NeRF with a hybrid tri-plane representation for efficiency.
2. **Connection to Previous:** Shifts from explicit voxels to continuous neural fields (NeRF) as 3D backbones, trading discretization artifacts for smooth geometry and fine detail.
3. **Outlook for Next:** Leads into hybrid and patch‑based backbones, and the role of diffusion losses for 3D (DreamFusion).

**Section 4 (Slides 19–…): Diffusion‑Based 3D Generation & Optimizing 3D with 2D Diffusion**

1. **Summary:** Introduces DreamFusion: use a pre‑trained 2D diffusion model (Imagen) as a “score” guidance to optimize a randomly initialized NeRF via score‑distillation sampling. No 3D training data required.
2. **Connection to Previous:** Combines NeRF backbones with diffusion‑model likelihoods—a departure from pure adversarial training to likelihood‑based 3D generation.
3. **Outlook for Next:** Suggests part‑based 3D generation and more advanced conditional/diffusion pipelines.

III. **Knowledge Connections**

* **From 2D to 3D:** Each chunk builds on 2D generative primitives (GANs, diffusion) by embedding an explicit or implicit 3D “backbone” module—voxels, radiance fields, tri‑planes—plus a differentiable renderer.
* **Prerequisites:** Familiarity with 2D GANs (StyleGAN/AdaIN), autoregressive/diffusion fundamentals, basic 3D rendering (volumetric rendering, rigid‑body transforms).
* **Future Expansion:** Later lectures will delve into part‑based 3D synthesis, conditional 3D from images/videos, and integration with real‑world 3D sensors.

IV. **Core Points & Memory Aids**

1. **3D Adversarial Nets:** Learn 3D via 2D renderings + differentiable renderer (HoloGAN).
   *Aid:* “Render‑and‑rogue GAN” → 3D features go first, then render.
2. **AdaIN in 3D:** Style mapping MLP → FiLM‑style AdaIN on 3D conv features.
   *Aid:* “Adaptive Instance Norm in 3D norm.”
3. **Neural Radiance Field (NeRF):** Continuous MLP → σ(x), c(x,d) → volume rendering.
   *Aid:* “SIREN sirens summon volume.”
4. **FiLM Conditioning + SIREN:** Style‑mapping yields γ, β for each sine layer → rich 3D detail.
   *Aid:* “Film the siren with style.”
5. **Progressive Growing:** Start low‑res; increase image resolution in discriminator to stabilize.
   *Aid:* “Grow from grass to granite.”
6. **Tri‑Plane Hybrid (EG3D):** Axis‑aligned 2D feature planes + small MLP → efficient NeRF.
   *Aid:* “Three planes sustain the NeRF.”
7. **Score Distillation Sampling:** Use 2D diffusion model as “critic” to optimize 3D NeRF in DreamFusion.
   *Aid:* “Distill the score into the 3D core.”
8. **Unsupervised 3D from 2D Images:** No 3D labels; leverage known camera distribution and renderer.
   *Aid:* “See 2D to free 3D.”

V. **Pre-Study Suggestions**

1. **Review:** How AdaIN and style mapping work in StyleGAN, especially FiLM‑based conditioning.
2. **Question:** What are the advantages/trade‑offs between voxel grids vs. implicit fields vs. mesh backbones?
3. **Explore:** Basic NeRF volume rendering equations and how density integrals produce pixel color.

VI. **Review Questions/Practice**

1. **Derive** the volumetric rendering integral and explain how discrete sampling approximates it.
2. **Implement** a minimal 3D GAN generator: start with a constant tensor, apply 3D convs, project and 2D convs.
3. **Experiment:** Optimize a simple NeRF using pre‑trained 2D diffusion score function and observe geometry convergence.
