#include "einsum.hpp"
#include <chrono>

int main() {
    // Define tensors I and D
    NDArray<int> I({ 100, 100, 100, 100 });
    NDArray<int> D({ 100, 100 });

    // Initialize tensor I with some formula, for example: I(p,q,r,s) = p + q + r + s
    auto start_init_I = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(4)
    for (size_t p = 0; p < 100; ++p) {
        for (size_t q = 0; q < 100; ++q) {
            for (size_t r = 0; r < 100; ++r) {
                for (size_t s = 0; s < 100; ++s) {
                    I({ p, q, r, s }) = p + q + r + s;
                }
            }
        }
    }
    auto end_init_I = std::chrono::high_resolution_clock::now();

    // Initialize tensor D with some formula, for example: D(r,s) = r + s
    auto start_init_D = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (size_t r = 0; r < 100; ++r) {
        for (size_t s = 0; s < 100; ++s) {
            D({ r, s }) = r + s;
        }
    }
    auto end_init_D = std::chrono::high_resolution_clock::now();

    // Use einsum for calculation
    auto start_einsum = std::chrono::high_resolution_clock::now();
    NDArray<int> J = einsum<int>("pqrs,rs->pq", { I, D });
    auto end_einsum = std::chrono::high_resolution_clock::now();

    // Print the result (printing such a large matrix might not be practical, so we just print a part of it)
    std::cout << "Result of einsum calculation (first 10x10 block):\n";
    for (size_t p = 0; p < 10; ++p) {
        for (size_t q = 0; q < 10; ++q) {
            std::cout << J({ p, q }) << " ";
        }
        std::cout << std::endl;
    }

    // Matrix multiplication using loops for comparison
    NDArray<int> J_loop({ 100, 100 });
    auto start_loop = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (size_t p = 0; p < 100; ++p) {
        for (size_t q = 0; q < 100; ++q) {
            int sum = 0;
            for (size_t r = 0; r < 100; ++r) {
                for (size_t s = 0; s < 100; ++s) {
                    sum += I({ p, q, r, s }) * D({ r, s });
                }
            }
            J_loop({ p, q }) = sum;
        }
    }
    auto end_loop = std::chrono::high_resolution_clock::now();

    // Print the result of the loop calculation (printing such a large matrix might not be practical, so we just print a part of it)
    std::cout << "Result of loop calculation (first 10x10 block):\n";
    for (size_t p = 0; p < 10; ++p) {
        for (size_t q = 0; q < 10; ++q) {
            std::cout << J_loop({ p, q }) << " ";
        }
        std::cout << std::endl;
    }

    // Output timing results
    auto duration_init_I = std::chrono::duration_cast<std::chrono::milliseconds>(end_init_I - start_init_I).count();
    auto duration_init_D = std::chrono::duration_cast<std::chrono::milliseconds>(end_init_D - start_init_D).count();
    auto duration_einsum = std::chrono::duration_cast<std::chrono::milliseconds>(end_einsum - start_einsum).count();
    auto duration_loop = std::chrono::duration_cast<std::chrono::milliseconds>(end_loop - start_loop).count();

    std::cout << "Initialization time for I: " << duration_init_I << " ms\n";
    std::cout << "Initialization time for D: " << duration_init_D << " ms\n";
    std::cout << "Einsum calculation time: " << duration_einsum << " ms\n";
    std::cout << "Loop calculation time: " << duration_loop << " ms\n";

    return 0;
}