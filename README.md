# Linear Algebra Concepts for AI

A comprehensive reference for core linear algebra concepts used in Artificial Intelligence and Machine Learning.

---

## 1. Scalars, Vectors, Matrices, and Tensors

- **Scalar**: A single number (e.g., learning rate, bias).  
- **Vector**: Ordered 1D array (e.g., feature vector of an image).  
- **Matrix**: 2D array (e.g., dataset with rows = samples, columns = features).  
- **Tensor**: Generalization to higher dimensions (e.g., RGB image → 3D tensor).  

---

## 2. Vector Operations

- **Addition & Subtraction**  
- **Scalar Multiplication**  
- **Dot Product (Inner Product)** → measures similarity (used in cosine similarity for NLP):  
  $$\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$$  
- **Cross Product** (only in 3D) → used in geometry and physics:  
  $$\mathbf{a} \times \mathbf{b} = 
  \begin{vmatrix}
  \mathbf{i} & \mathbf{j} & \mathbf{k} \\
  a_1 & a_2 & a_3 \\
  b_1 & b_2 & b_3
  \end{vmatrix}$$  

---

## 3. Matrix Operations

- **Addition, Subtraction, Scalar Multiplication**  
- **Matrix Multiplication** → core operation in neural networks:  
  $$C = A \cdot B$$  
- **Transpose** →  
  $$A^T$$  
- **Identity Matrix** ($I$) → like multiplying by 1  
- **Inverse Matrix** ($A^{-1}$) → solving linear systems  
- **Determinant** → volume, singularity check:  
  $$\det(A)$$  
- **Normalization** → scaling rows/columns to unit norm, often applied before feeding data to models  

---

## 4. Systems of Linear Equations

- Represented as:  
  $$Ax = b$$  
- Solutions: unique, infinite, or none  
- Solved with matrix inverses or decomposition  

---

## 5. Linear Transformations

- Matrix as a transformation (e.g., rotation, scaling, projection)  
- Crucial for understanding how data moves through neural networks  

---

## 6. Eigenvalues and Eigenvectors

- $$Av = \lambda v$$  
- **Eigenvectors** = directions of variance  
- **Eigenvalues** = magnitude of variance along the eigenvector  
- Used in **PCA (Principal Component Analysis)**, spectral clustering, etc.  

---

## 7. Matrix Decompositions

- **LU Decomposition** → efficient solving of linear systems  
- **QR Decomposition** → orthogonality  
- **Singular Value Decomposition (SVD)** → foundation of PCA, used in recommendation systems (e.g., Netflix, YouTube)  

---

## 8. Orthogonality & Projections

- **Orthogonal vectors** → dot product = 0:  
  $$\mathbf{u} \cdot \mathbf{v} = 0$$  
- **Orthonormal basis** → like standard axes in 3D (x, y, z)  
- **Projection** → mapping data onto lower dimensions:  
  $$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}$$  
- Key in **dimensionality reduction**  

---

## 9. Vector Spaces & Subspaces

- **Span, Basis, Rank, Null space**  
- **Rank** = number of independent features:  
  $$\text{rank}(A)$$  
- **Null space** = solutions to:  
  $$Ax = 0$$  

---

## 10. Special Matrices

- **Diagonal** → simple scaling  
- **Symmetric** → real eigenvalues  
- **Orthogonal** → preserves length & angle  
- **Sparse** → many zeros, used in NLP  

---

## 11. Norms & Distance Measures

- **Euclidean Distance (L2)**:  
  $$\|x - y\|_2 = \sqrt{\sum_i (x_i - y_i)^2}$$  
- **Manhattan Distance (L1)**:  
  $$\|x - y\|_1 = \sum_i |x_i - y_i|$$  
- **Cosine Similarity** → crucial for NLP embeddings:  
  $$\text{cosine\_sim}(x, y) = \frac{x \cdot y}{\|x\| 2 \|y\| 2}$$


### References

- Book: **Discrete Mathematics and Its Applications (7th Edition)** – [`./books/Discrete_Mathematics_and_Its_Applications_(7th_Edition).pdf`](./books/Discrete_Mathematics_and_Its_Applications_(7th_Edition).pdf)  
