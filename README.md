# Linear Algebra Concepts for AI

A concise reference guide to the core linear algebra concepts that power **Artificial Intelligence (AI)** and **Machine Learning (ML)**.

---

## 1. Scalars, Vectors, Matrices, and Tensors

- **Scalar** → a single number (e.g., learning rate, bias).  
- **Vector** → ordered 1D array (e.g., feature vector of an image).  
- **Matrix** → 2D array (e.g., dataset with rows = samples, columns = features).  
- **Tensor** → generalization to higher dimensions (e.g., RGB image → 3D tensor).  

---

## 2. Vector Operations

- **Addition & Subtraction**  
- **Scalar Multiplication**  
- **Dot Product (Inner Product)** → measures similarity:  

  $$
  \mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i
  $$

- **Cross Product** (only in 3D):  

  $$
  \mathbf{a} \times \mathbf{b} = 
  \begin{vmatrix}
  \mathbf{i} & \mathbf{j} & \mathbf{k} \\
  a_1 & a_2 & a_3 \\
  b_1 & b_2 & b_3
  \end{vmatrix}
  $$

---

## 3. Matrix Operations

- **Addition, Subtraction, Scalar Multiplication**  
- **Matrix Multiplication** → key in neural networks:  

  $$
  C = A \cdot B
  $$

- **Transpose**: $A^T$  
- **Identity Matrix**: $I$  
- **Inverse Matrix**: $A^{-1}$ (solves linear systems)  
- **Determinant**: $\det(A)$ → volume, singularity check  
- **Normalization** → scaling rows/columns to unit norm before model input  

---

## 4. Norms & Distance Measures

- **Euclidean Distance (L2):**

  $$
  \|x - y\|_2 = \sqrt{\sum_i (x_i - y_i)^2}
  $$

- **Manhattan Distance (L1):**

  $$
  \|x - y\|_1 = \sum_i |x_i - y_i|
  $$

- **Cosine Similarity:**

  ![Cosine similarity](https://latex.codecogs.com/svg.latex?\color{white}\cos(\theta)=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\;\|\mathbf{b}\|})

---

## 5. Systems of Linear Equations

- General form: $Ax = b$  
- Solutions: unique, infinite, or none  
- Solved using inverses or decompositions  

---

## 6. Linear Transformations

- Matrices as transformations (rotation, scaling, projection)  
- Crucial for understanding how data flows through neural networks  

---

## 7. Eigenvalues and Eigenvectors

- Definition: $Av = \lambda v$  
- **Eigenvectors** → directions of variance  
- **Eigenvalues** → magnitude along eigenvectors  
- Applications: PCA, spectral clustering, dimensionality reduction  

---

## 8. Matrix Decompositions

- **LU Decomposition** → efficient linear solving  
- **QR Decomposition** → orthogonal basis  
- **Singular Value Decomposition (SVD)** → foundation of PCA, used in recommender systems  

---

## 9. Orthogonality & Projections

- **Orthogonal vectors**: $\mathbf{u} \cdot \mathbf{v} = 0$  
- **Orthonormal basis** → like standard axes in 3D  
- **Projection**:  

  $$
  \text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v}
  $$

- Used in dimensionality reduction  

---

## 10. Vector Spaces & Subspaces

- **Span, Basis, Rank, Null Space**  
- Rank: $\text{rank}(A)$ = number of independent features  
- Null space: solutions to $Ax = 0$  

---

## 11. Special Matrices

- **Diagonal** → scaling  
- **Symmetric** → real eigenvalues  
- **Orthogonal** → preserves length & angle  
- **Sparse** → many zeros (e.g., NLP applications)  

---

### References

- *Discrete Mathematics and Its Applications (7th Edition)*  
  [`./books/Discrete_Mathematics_and_Its_Applications_(7th_Edition).pdf`](./books/Discrete_Mathematics_and_Its_Applications_(7th_Edition).pdf)
