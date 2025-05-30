

# 3DVC 2025 Project

## 3D VISUAL COMPUTING  
## COURSE PROJECT  
## 2025 Spring

---

### OVERVIEW

In this project, we provide two sub-directions for you to explore, and you will need to select one of them to complete. By the end of the project, you are required to submit a PDF report detailing your methodology, experiments, and findings, along with a demo and the full source code. Please note that directly using pre-trained foundation models such as Segment-Anything as a component of your component or for demo purpose is not allowed.

---

### 1. CATEGORY-LEVEL OBJECT POSE ESTIMATION

#### 1.1 DESCRIPTION

Object pose estimation involves predicting the 6D pose (rotation and translation) and scale of an object from an RGB-D image or a depth image. Existing approaches fall into two categories: instance-level and category-level pose estimation. In instance-level estimation, the object’s CAD model is assumed to be known. The typical pipeline first predicts an object mask from the input image, extracts the object’s point cloud from the segmented RGB-D data, and then estimates the pose using alignment algorithms like ICP (Iterative Closest Point) (Besl & McKay, 1992). However, this reliance on a known CAD model limits real-world applicability, as humans can intuitively infer poses for novel objects without such prior knowledge. This motivates category-level pose estimation, where the goal is to predict poses that align objects—including unseen instances—to a category-specific canonical space using only a collection of RGB-D images from the same category. Crucially, the network must generalize to novel objects within the category, enabling pose estimation without CAD models.

This project focuses on investigating category-level object pose estimation. You will: (1) study existing algorithms, (2) identify potential limitations and improvement opportunities, and (3) explore extensions to advance current approaches.

#### 1.2 REQUIREMENTS

##### 1.2.1 BASIC REQUIREMENTS

You will begin by studying category-level object pose estimation (Wang et al., 2019) and instance-level approaches (Rad & Lepetit, 2017; Tekin et al., 2018; Kehl et al., 2017; Xiang et al., 2017; Tejani et al., 2017). The primary project objective involves implementing the Normalized Object Coordinate Space (NOCS) framework in PyTorch, conducting experiments, and analyzing results. Note that while an official TensorFlow and Keras implementation is available, your implementation must use PyTorch. Please conduct experiments using the dataset from NOCS, which is available in the official NOCS GitHub repository.

#### 1.3 ALGORITHM DESIGN

Enhance your algorithm’s performance, compare the improved version against the baseline and highlight its advantages, such as stronger generalization on the real test split. Your report should clearly describe your modification, explain the intuition behind its effectiveness, and provide experimental results to demonstrate its superiority over the original approach.

#### 1.4 ARTICULATED OBJECT POSE ESTIMATION

Can you extend your algorithm to articulated object pose estimation, building upon the normalized space concept proposed in NOCS? Unlike rigid objects, which are treated as a single part, articulated objects (e.g., eyeglasses) consist of multiple rigid components connected by joints. Extending NOCS to such cases would require defining a normalized space for each individual part while also estimating part segmentations.

Articulated-pose (Li et al., 2020) extends NOCS to articulated objects and can serve as a reference for your implementation. You may adopt their experimental setup and evaluation protocol for your work. Additionally, the official codebase and the PyTorch version have been publicly released, which you can consult for guidance. However, as emphasized in the assignment, direct copying of code or results is not allowed. For articulated pose estimation, please conduct experiments using the dataset released in ANCSH-pytorch (eyeglasses and onedoor category).

#### 1.5 GRADING

A reference grading policy for this project:

- **(60 pts)** Present the results of your NOCS implementation, detailing your efforts to reproduce the performance reported in the original paper. Even if your current results are not yet satisfactory, you will receive credit for demonstrating a clear understanding of the method and documenting your attempts to match the original performance (e.g., hyperparameter tuning, architectural adjustments, or data preprocessing). Please include both quantitative metrics and qualitative analysis.

- **(80 pts)** Successfully implement NOCS and achieve satisfactory performance matching the original benchmarks. Conduct comprehensive experiments using different backbone architectures and provide detailed comparisons of their performance.

- **(90 pts)** Option 1: Propose and implement your own improvements to enhance NOCS performance. Option 2: Extend NOCS to articulated pose estimation. You will receive full credit for this section as long as you demonstrate thoughtful effort—even if your proposed improvement or articulated object implementation does not achieve optimal results.

- **(100 pts)** Option 1: Make a noticeable improvement in NOCS. Option 2: Demonstrate a workable solution for articulated object pose estimation.

---

### 2. 3D GENERATION

#### 2.1 DESCRIPTION

3D generation has emerged as a prominent research area, attracting significant attention due to its wide-ranging applications (Shue et al., 2023; Gupta et al., 2023; Luo & Hu, 2021; Kalischek et al., 2022; Liu et al., 2023; Chou et al., 2023; Vahdat et al., 2022; Chen et al., 2024; Yan et al., 2024; Siddiqui et al., 2024; Tang et al., 2023). Recent advances have produced diverse models capable of generating 3D assets across multiple representations, including point clouds, meshes, implicit SDFs, triplanes, and novel approaches like geometric primitives. This project invites you to investigate these techniques, explore potential enhancements, and examine their practical applications.

#### 2.2 REQUIREMENTS

##### 2.2.1 BASIC REQUIREMENTS

Familiarize yourself with diffusion models (Ho et al., 2020; Song et al., 2020) and their extensions to 3D content generation (Luo & Hu, 2021; Kalischek et al., 2022; Liu et al., 2023; Chou et al., 2023).

The basic requirements of this project consist of two parts: 1) Literature Review. Conduct a comprehensive review of diffusion models for 3D generation. A curated collection of relevant works can be found in this repository. 2) Implementation an unconditional 3D generation method. Select one unconditional 3D generation method from existing works that employs your favorite representation (e.g., point clouds, meshes, or implicit fields). Reproduce their results either by implementing the approach independently, or utilizing the authors’ publicly available code. Please download dataset for this project from here. You should conduct experiments at least on four categories, that is airplane (02691156), bag (02773838), table (04379243), and car (02958343).

##### 2.2.2 ALGORITHM DESIGN

Improve the performance of your baseline method and conduct comparative evaluations between your enhanced algorithm and the original baseline. Your report should include: (1) a clear description of your proposed improvement and (2) experimental results demonstrating performance gains.

##### 2.2.3 APPLICATIONS

Extend your model to support one of the following conditional generation tasks:

- **Text-to-3D**: Develop a pipeline that generates 3D assets whose geometry and semantics align with given text descriptions.

- **Image-to-3D**: Create a system that produces 3D representations faithful to the content of input 2D images.

- **Other Interesting Applications**: Propose and implement an original conditional generation task. Note: If building upon existing unconditional generation work, your conditional extension must differ from any conditional pipelines already included in the base method. You are expected to develop a new application scenario not previously explored in the original work.

#### 2.3 ADDITIONAL COMMENTS

- Given the substantial effort required to implement diffusion models for 3D content generation from scratch, you can use open-source implementations as your foundation. We recommend: 1) Reviewing both the paper and code repository of potential base implementations; 2) Selecting a good project (e.g., high GitHub stars); 3) Setting up the environment and testing example/inference pipelines. After these steps, you may get a sense of whether you are confident enough to build your project based on the repo. Pick the one that you feel most comfortable with.

- Conditional 3D generation is a popular topic with a large amount of reference research works. To extend your unconditional pipeline to conditional generation, try to make yourself familiar with related literature and borrow their techniques to your project.

#### 2.4 GRADING

A reference grading policy for this project is as follows:

- **Basic Implementation (60 points)**: Successful implementation and reporting of baseline unconditional 3D generation results, including quantitative metrics and qualitative analysis.

- **Extension Attempt (80 points)**: Demonstrate either:
  - Meaningful attempts to improve the baseline method’s performance, or
  - Valid efforts to extend the model for conditional generation applications

Note: Full credit for this section only requires documented effort and analysis, even if the attempts don’t fully succeed.

- **Advanced Achievement (100 points)**: Deliver either:
  - A successful, measurable improvement over the baseline method, or
  - A fancy application extension.

---

### REFERENCES

Paul J Besl and Neil D McKay. Method for registration of 3-d shapes. In Sensor fusion IV: control paradigms and data structures, volume 1611, pp. 586–606. Spie, 1992.

Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. arXiv preprint arXiv:2409.12957, 2024.

Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2262–2272, 2023.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Nikolai Kalischek, Torben Peters, Jan D Wegner, and Konrad Schindler. Tetradiffusion: Tetrahedral diffusion models for 3d shape generation. arXiv preprint arXiv:2211.13220, 2022.

Wadim Kehl, Fabian Manhardt, Federico Tombari, Slobodan Ilic, and Nassir Navab. Ssd-6d: Making rgb-based 3d detection and 6d pose estimation great again. In Proceedings of the IEEE international conference on computer vision, pp. 1521–1529, 2017.

Xiaolong Li, He Wang, Li Yi, Leonidas J Guibas, A Lynn Abbott, and Shuran Song. Category-level articulated object pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3706–3715, 2020.

Zhen Liu, Yao Feng, Michael J Black, Derek Nowrouzezahrai, Liam Paull, and Weiyang Liu. Meshdiffusion: Score-based generative 3d mesh modeling. arXiv preprint arXiv:2303.08133, 2023.

Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2837–2845, 2021.

Mahdi Rad and Vincent Lepetit. Bb8: A scalable, accurate, robust to partial occlusion method for predicting the 3d poses of challenging objects without using depth. In Proceedings of the IEEE international conference on computer vision, pp. 3828–3836, 2017.

J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20875–20886, 2023.

Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner. Meshgpt: Generating triangle meshes with decoder-only transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19615–19625, 2024.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Zhicong Tang, Shuyang Gu, Chunyu Wang, Ting Zhang, Jianmin Bao, Dong Chen, and Baining Guo. Volumediffusion: Flexible text-to-3d generation with efficient volumetric encoder. arXiv preprint arXiv:2312.11459, 2023.

Alykhan Tejani, Rigas Kouskouridas, Andreas Doumanoglou, Danhang Tang, and Tae-Kyun Kim. Latent-class hough forests for 6 dof object pose estimation. IEEE transactions on pattern analysis and machine intelligence, 40(1):119–132, 2017.

Bugra Tekin, Sudipta N Sinha, and Pascal Fua. Real-time seamless single shot 6d object pose prediction. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 292–301, 2018.

Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, Karsten Kreis, et al. Lion: Latent point diffusion models for 3d shape generation. Advances in Neural Information Processing Systems, 35:10021–10039, 2022.

He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2642–2651, 2019.

Yu Xiang, Tanner Schmidt, Venkatraman Narayanan, and Dieter Fox. Posecnn: A convolutional neural network for 6d object pose estimation in cluttered scenes. arXiv preprint arXiv:1711.00199, 2017.

Xingguang Yan, Han-Hung Lee, Ziyu Wan, and Angel X Chang. An object is worth 64x64 pixels: Generating 3d object via image diffusion. arXiv preprint arXiv:2408.03178, 2024.