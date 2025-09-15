
import cupy as cp
import scipy as sp

print("=== Linear Algebra Concepts on GPU using cuTENSOR/cuBLAS (via CuPy) ===\n")

# 1. Scalars, Vectors, Matrices, Tensors
scalar = cp.array(3.0)  # scalar
vector = cp.array([1.0, 2.0, 3.0])  # vector
matrix = cp.array([[1,2,3],[4,5,6],[7,8,9]], dtype=cp.float32)  # 3x3 matrix
tensor = cp.ones((3,3,3), dtype=cp.float32)  # 3D tensor

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix:\n", matrix)
print("Tensor shape:", tensor.shape, "\n")

# 2. Vector operations
v1 = cp.array([1,2,3], dtype=cp.float32)
v2 = cp.array([4,5,6], dtype=cp.float32)

v_add = v1 + v2
v_sub = v1 - v2
v_scalar_mul = 2 * v1
v_dot = cp.dot(v1, v2)
v_cross = cp.cross(v1, v2)

print("Vector Addition:", v_add)
print("Vector Subtraction:", v_sub)
print("Scalar Multiplication:", v_scalar_mul)
print("Dot Product:", v_dot)
print("Cross Product:", v_cross, "\n")

# 3. Matrix Operations
A = cp.array([[1,2],[3,4]], dtype=cp.float32)
B = cp.array([[5,6],[7,8]], dtype=cp.float32)

C_add = A + B
C_sub = A - B
C_scalar = 2 * A
C_mult = cp.matmul(A,B)
C_transpose = A.T
C_identity = cp.eye(2, dtype=cp.float32)
C_inverse = cp.linalg.inv(A)
C_det = cp.linalg.det(A)
C_normalized = A / cp.linalg.norm(A, axis=0)  # normalize columns

print("Matrix Addition:\n", C_add)
print("Matrix Subtraction:\n", C_sub)
print("Matrix Scalar Multiplication:\n", C_scalar)
print("Matrix Multiplication:\n", C_mult)
print("Matrix Transpose:\n", C_transpose)
print("Identity Matrix:\n", C_identity)
print("Inverse Matrix:\n", C_inverse)
print("Determinant:", C_det)
print("Normalized Matrix (columns):\n", C_normalized, "\n")

# 4. Norms & Distance Measures
x = cp.array([1,2,3], dtype=cp.float32)
y = cp.array([4,5,6], dtype=cp.float32)

euclidean = cp.linalg.norm(x-y)
manhattan = cp.linalg.norm(x-y, ord=1)
cosine_sim = cp.dot(x,y) / (cp.linalg.norm(x) * cp.linalg.norm(y))

print("Euclidean Distance:", euclidean)
print("Manhattan Distance:", manhattan)
print("Cosine Similarity:", cosine_sim, "\n")

# 5. Systems of Linear Equations
A_sys = cp.array([[3,1],[1,2]], dtype=cp.float32)
b_sys = cp.array([9,8], dtype=cp.float32)
x_sys = cp.linalg.solve(A_sys, b_sys)

print("Solution of Ax=b:", x_sys, "\n")

# 6. Linear Transformations
vec = cp.array([1,0], dtype=cp.float32)
rot_matrix = cp.array([[0, -1],[1,0]], dtype=cp.float32)  # 90 degree rotation
vec_rotated = cp.matmul(rot_matrix, vec)
print("Linear Transformation (Rotation) of vector:", vec_rotated, "\n")

# 7. Eigenvalues and Eigenvectors
eig_vals, eig_vecs = cp.linalg.eigh(cp.array([[2,0],[0,3]], dtype=cp.float32))
print("Eigenvalues:", eig_vals)
print("Eigenvectors:\n", eig_vecs, "\n")

# 8. Matrix Decompositions
# QR decomposition
Q, R = cp.linalg.qr(cp.array([[1,2],[3,4]], dtype=cp.float32))
print("QR Decomposition Q:\n", Q)
print("QR Decomposition R:\n", R)

# SVD
U_svd, S_svd, Vh_svd = cp.linalg.svd(cp.array([[1,2],[3,4]], dtype=cp.float32))
print("SVD U:\n", U_svd)
print("SVD S:\n", S_svd)
print("SVD V^T:\n", Vh_svd, "\n")

# 9. Orthogonality & Projections
u = cp.array([1,0], dtype=cp.float32)
v = cp.array([0,1], dtype=cp.float32)
dot_uv = cp.dot(u,v)
proj_u_on_v = (cp.dot(u,v)/cp.dot(v,v))*v
print("Dot Product of orthogonal vectors:", dot_uv)
print("Projection of u onto v:", proj_u_on_v, "\n")

# 10. Vector Spaces & Subspaces
A_sub = cp.array([[1,2,3],[0,1,4],[1,3,7]], dtype=cp.float32)
rank_A = cp.linalg.matrix_rank(A_sub)
null_space = sp.linalg.null_space(A_sub.get())  # get() brings data to CPU for null space
print("Rank of A:", rank_A)
print("Null space of A:\n", null_space, "\n")

# 11. Special Matrices
diag_matrix = cp.diag(cp.array([1,2,3], dtype=cp.float32))
sym_matrix = cp.array([[2,1],[1,2]], dtype=cp.float32)
orth_matrix = cp.array([[1,0],[0,-1]], dtype=cp.float32)
sparse_matrix = cp.array([[0,0,3],[0,0,0],[4,0,0]], dtype=cp.float32)

print("Diagonal Matrix:\n", diag_matrix)
print("Symmetric Matrix:\n", sym_matrix)
print("Orthogonal Matrix:\n", orth_matrix)
print("Sparse Matrix:\n", sparse_matrix)
