#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <stdexcept>
#include <functional>

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
};

// 新的辅助函数，用于解析下标字符串
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

template<typename T>
void einsumHelper(const std::vector<std::string>& inputSubscripts,
    const std::string& outputSubscripts,
    const std::vector<NDArray<T>>& tensors,
    std::vector<size_t>& currentIndices,
    NDArray<T>& result,
    size_t tensorIndex = 0,
    T currentProduct = T(1)) {

    if (tensorIndex == tensors.size()) {
        std::vector<size_t> outputIndices(result.getShape().size(), 0);
        for (size_t i = 0; i < outputSubscripts.size(); ++i) {
            char c = outputSubscripts[i];
            for (size_t j = 0; j < inputSubscripts.size(); ++j) {
                size_t pos = inputSubscripts[j].find(c);
                if (pos != std::string::npos) {
                    outputIndices[i] = currentIndices[j * inputSubscripts[j].size() + pos];
                    break;
                }
            }
        }
        result(outputIndices) += currentProduct;
        return;
    }

    const auto& tensor = tensors[tensorIndex];
    const auto& subscript = inputSubscripts[tensorIndex];
    std::vector<size_t> indices(subscript.size());

    std::function<void(size_t)> recurse = [&](size_t dim) {
        if (dim == subscript.size()) {
            T value = tensor(indices);
            einsumHelper(inputSubscripts, outputSubscripts, tensors, currentIndices, result,
                tensorIndex + 1, currentProduct * value);
            return;
        }
        for (size_t i = 0; i < tensor.getShape()[dim]; ++i) {
            indices[dim] = i;
            currentIndices[tensorIndex * subscript.size() + dim] = i;
            recurse(dim + 1);
        }
        };

    recurse(0);
}

// 修改后的einsum函数
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
    std::vector<size_t> currentIndices(std::accumulate(inputSubscripts.begin(), inputSubscripts.end(), 0,
        [](size_t sum, const std::string& s) { return sum + s.size(); }));

    einsumHelper(inputSubscripts, outputSubscripts, tensors, currentIndices, result);
    return result;
}

