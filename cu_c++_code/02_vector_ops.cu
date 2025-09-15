#include <iostream>
#include <vector>
#include <cmath>

// --- Helper: print vector ---
void printVector(const std::vector<float>& v, const std::string& name) {
    std::cout << name << " = [ ";
    for (auto x : v) std::cout << x << " ";
    std::cout << "]" << std::endl;
}

// --- Vector Addition ---
std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) result[i] = a[i] + b[i];
    return result;
}

// --- Vector Subtraction ---
std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) result[i] = a[i] - b[i];
    return result;
}

// --- Scalar Multiplication ---
std::vector<float> scalarMultiply(const std::vector<float>& a, float s) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) result[i] = a[i] * s;
    return result;
}

// --- Dot Product ---
float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); i++) result += a[i] * b[i];
    return result;
}

// --- Cross Product (3D only) ---
std::vector<float> cross(const std::vector<float>& a, const std::vector<float>& b) {
    return {
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    };
}

// --- L1 Norm (Manhattan) ---
float l1Norm(const std::vector<float>& a) {
    float sum = 0.0f;
    for (auto x : a) sum += std::fabs(x);
    return sum;
}

// --- L2 Norm (Euclidean) ---
float l2Norm(const std::vector<float>& a) {
    float sum = 0.0f;
    for (auto x : a) sum += x * x;
    return std::sqrt(sum);
}

int main() {
    // Example vectors
    std::vector<float> v1 = {1, 2, 3};
    std::vector<float> v2 = {4, 5, 6};

    printVector(v1, "v1");
    printVector(v2, "v2");

    // Addition & Subtraction
    printVector(add(v1, v2), "v1 + v2");
    printVector(subtract(v1, v2), "v1 - v2");

    // Scalar multiplication
    printVector(scalarMultiply(v1, 2.0f), "2 * v1");

    // Dot product
    std::cout << "Dot(v1, v2) = " << dot(v1, v2) << std::endl;

    // Cross product (3D only)
    printVector(cross(v1, v2), "v1 x v2");

    // Norms
    std::cout << "L1 norm of v1 = " << l1Norm(v1) << std::endl;
    std::cout << "L2 norm of v1 = " << l2Norm(v1) << std::endl;

    return 0;
}
