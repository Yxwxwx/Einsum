#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <set>
#include <sstream>
#include <omp.h>

template<typename T>
class NDArray {
public:
    NDArray(const std::vector<size_t>& shape) : shape(shape) {
        size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        data.resize(totalSize);
    }

    T& operator()(const std::vector<size_t>& indices) {
        return data[calculateIndex(indices)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data[calculateIndex(indices)];
    }

    const std::vector<size_t>& getShape() const {
        return shape;
    }

    std::vector<T>& getData() {
        return data;
    }

    void print() const {
        printRecursive(0, std::vector<size_t>(), true);
        std::cout << "\n";
    }

private:
    std::vector<T> data;
    std::vector<size_t> shape;

    size_t calculateIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Number of indices does not match array dimensions");
        }
        size_t index = 0;
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }

    void printRecursive(size_t dim, std::vector<size_t> indices, bool isOutermost) const {
        if (dim == shape.size()) {
            std::cout << (*this)(indices);
        } else {
            std::cout << "[";
            for (size_t i = 0; i < shape[dim]; ++i) {
                indices.push_back(i);
                printRecursive(dim + 1, indices, false);
                indices.pop_back();
                if (i < shape[dim] - 1) {
                    std::cout << " ";
                }
            }
            std::cout << "]";
        }
    }
};

std::vector<std::string> parseSubscripts(const std::string& subscripts) {
    std::vector<std::string> result;
    size_t start = 0, end = 0;
    while ((end = subscripts.find(',', start)) != std::string::npos) {
        result.push_back(subscripts.substr(start, end - start));
        start = end + 1;
    }
    result.push_back(subscripts.substr(start));
    return result;
}

std::string currentIndicesToString(const std::map<char, size_t>& currentIndices) {
    std::ostringstream oss;
    for (const auto& pair : currentIndices) {
        oss << pair.first << ":" << pair.second << " ";
    }
    return oss.str();
}

template<typename T>
void einsumHelper(const std::vector<std::string>& inputSubscripts,
    const std::string& outputSubscripts,
    const std::vector<NDArray<T>>& tensors,
    std::map<char, size_t>& currentIndices,
    NDArray<T>& result,
    const std::map<char, size_t>& dimSizes) {

    std::vector<std::map<char, size_t>::iterator> iterators;
    for (auto it = currentIndices.begin(); it != currentIndices.end(); ++it) {
        iterators.push_back(it);
    }

    // Calculate the total number of iterations
    size_t totalIterations = 1;
    for (const auto& dim : dimSizes) {
        totalIterations *= dim.second;
    }

    #pragma omp parallel for
    for (size_t iter = 0; iter < totalIterations; ++iter) {
        std::map<char, size_t> localIndices = currentIndices;

        // Determine the current indices based on the iteration number
        size_t tempIter = iter;
        for (int i = iterators.size() - 1; i >= 0; --i) {
            char currentAxis = iterators[i]->first;
            size_t size = dimSizes.at(currentAxis);
            localIndices[currentAxis] = tempIter % size;
            tempIter /= size;
        }

        T product = 1;
        for (size_t i = 0; i < tensors.size(); ++i) {
            std::vector<size_t> indices;
            for (char c : inputSubscripts[i]) {
                indices.push_back(localIndices[c]);
            }
            product *= tensors[i](indices);
        }

        std::vector<size_t> outputIndices;
        for (char c : outputSubscripts) {
            outputIndices.push_back(localIndices[c]);
        }

        #pragma omp critical
        {
            result(outputIndices) += product;
        }
    }
}

template<typename T>
NDArray<T> einsum(const std::string& subscripts, const std::vector<NDArray<T>>& tensors) {
    size_t arrowPos = subscripts.find("->");
    if (arrowPos == std::string::npos) {
        throw std::invalid_argument("Invalid subscripts string: missing '->'");
    }

    std::string inputSubscriptsStr = subscripts.substr(0, arrowPos);
    std::string outputSubscripts = subscripts.substr(arrowPos + 2);

    std::vector<std::string> inputSubscripts = parseSubscripts(inputSubscriptsStr);
    if (inputSubscripts.size() != tensors.size()) {
        throw std::invalid_argument("Number of input subscripts does not match number of tensors");
    }

    std::map<char, size_t> dimSizes;
    for (size_t i = 0; i < inputSubscripts.size(); ++i) {
        const auto& subscript = inputSubscripts[i];
        const auto& tensor = tensors[i];
        if (subscript.size() != tensor.getShape().size()) {
            throw std::invalid_argument("Subscript does not match tensor dimensions");
        }
        for (size_t j = 0; j < subscript.size(); ++j) {
            dimSizes[subscript[j]] = tensor.getShape()[j];
        }
    }

    std::vector<size_t> resultShape;
    for (char c : outputSubscripts) {
        if (dimSizes.find(c) == dimSizes.end()) {
            throw std::invalid_argument("Output subscript not found in input subscripts");
        }
        resultShape.push_back(dimSizes[c]);
    }

    NDArray<T> result(resultShape);
    std::fill(result.getData().begin(), result.getData().end(), T(0));

    std::map<char, size_t> currentIndices;
    for (const auto& subscript : inputSubscripts) {
        for (char c : subscript) {
            currentIndices[c] = 0;
        }
    }

    einsumHelper(inputSubscripts, outputSubscripts, tensors, currentIndices, result, dimSizes);

    return result;
}