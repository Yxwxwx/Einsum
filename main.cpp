#include "einsum.h"

int main() {
    std::vector<size_t> shape = {2, 2};
    NDArray<double> A(shape);
    NDArray<double> B(shape);

    A({0, 0}) = 1; A({0, 1}) = 2;
    A({1, 0}) = 3; A({1, 1}) = 4;

    B({0, 0}) = 5; B({0, 1}) = 6;
    B({1, 0}) = 7; B({1, 1}) = 8;

    std::vector<NDArray<double>> tensors = {A, B};

    // 计算矩阵乘法 C[i,k] = A[i,j] * B[j,k]
    try {
        NDArray<double> result = einsum("ij,jk->ik", tensors);

        std::cout << "Result of einsum('ij,jk->ik'):" << std::endl;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 2; ++j) {
                std::cout << result({i, j}) << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}