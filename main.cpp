#include "einsum.hpp"

int main() {
    // Define tensors I and D
    NDArray<int> I({ 2, 2, 2, 2 });
    I({ 0, 0, 0, 0 }) = 1; I({ 0, 0, 0, 1 }) = 2;
    I({ 0, 0, 1, 0 }) = 3; I({ 0, 0, 1, 1 }) = 4;
    I({ 0, 1, 0, 0 }) = 5; I({ 0, 1, 0, 1 }) = 6;
    I({ 0, 1, 1, 0 }) = 7; I({ 0, 1, 1, 1 }) = 8;
    I({ 1, 0, 0, 0 }) = 9; I({ 1, 0, 0, 1 }) = 10;
    I({ 1, 0, 1, 0 }) = 11; I({ 1, 0, 1, 1 }) = 12;
    I({ 1, 1, 0, 0 }) = 13; I({ 1, 1, 0, 1 }) = 14;
    I({ 1, 1, 1, 0 }) = 15; I({ 1, 1, 1, 1 }) = 16;

    NDArray<int> D({ 2, 2 });
    D({ 0, 0 }) = 1; D({ 0, 1 }) = 2;
    D({ 1, 0 }) = 3; D({ 1, 1 }) = 4;
    // Use einsum for calculation
    NDArray<int> J = einsum<int>("pqrs,rs->pq", { I, D });

    // Print the result
    std::cout << "Result of einsum calculation:\n";
    J.print();

    // Matrix multiplication using loops for comparison
    NDArray<int> J_loop({ 2, 2 });
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            int sum = 0;
            for (size_t r = 0; r < 2; ++r) {
                for (size_t s = 0; s < 2; ++s) {
                    sum += I({ i, j, r, s }) * D({ r, s });
                }
            }
            J_loop({ i, j }) = sum;
        }
    }

    // Print the result of the loop calculation
    std::cout << "Result of loop calculation:\n";
    J_loop.print();

    return 0;
}