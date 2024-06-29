#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <stdexcept>
#include <functional>

// 定义多维数组类
template<typename T>
class NDArray {
public:
    NDArray(const std::vector<size_t>& shape) : shape(shape) {
        size_t totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        data.resize(totalSize);
    }

    T& operator()(const std::vector<size_t>& indices) {
        return data[calculateIndex(shape, indices)];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        return data[calculateIndex(shape, indices)];
    }

    const std::vector<size_t>& getShape() const {
        return shape;
    }

private:
    std::vector<T> data;
    std::vector<size_t> shape;

    size_t calculateIndex(const std::vector<size_t>& shape, const std::vector<size_t>& indices) const {
        size_t index = 0;
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            index += indices[i] * stride;
            stride *= shape[i];
        }
        return index;
    }
};

// 辅助函数，递归遍历所有可能的索引
template<typename T>
void einsumHelper(const std::string& subscripts, const std::vector<NDArray<T>>& tensors,
                  std::vector<size_t>& indices, std::vector<size_t>& resultIndices,
                  NDArray<T>& result, const std::map<char, size_t>& dimMap, size_t dim, T currentProduct) {
    if (dim == indices.size()) {
        result(resultIndices) += currentProduct;
        return;
    }

    char subscript = subscripts[dim];
    size_t tensorIndex = dimMap.at(subscript);
    size_t maxIndex = tensors[tensorIndex].getShape()[dim];

    for (size_t i = 0; i < maxIndex; ++i) {
        indices[dim] = i;
        if (dim < resultIndices.size() && dimMap.find(subscript) != dimMap.end() && dimMap.at(subscript) == dim) {
            resultIndices[dimMap.at(subscript)] = i;
        }
        einsumHelper(subscripts, tensors, indices, resultIndices, result, dimMap, dim + 1, currentProduct * tensors[tensorIndex](indices));
    }
}

// 爱因斯坦求和函数
template<typename T>
NDArray<T> einsum(const std::string& subscripts, const std::vector<NDArray<T>>& tensors) {
    size_t arrowPos = subscripts.find("->");
    if (arrowPos == std::string::npos) {
        throw std::invalid_argument("Invalid subscripts string: missing '->'");
    }
    
    std::string inputSubscripts = subscripts.substr(0, arrowPos);
    std::string outputSubscripts = subscripts.substr(arrowPos + 2);

    std::cout << "Input subscripts: " << inputSubscripts << std::endl;
    std::cout << "Output subscripts: " << outputSubscripts << std::endl;

    std::map<char, size_t> dimMap;
    size_t tensorCount = 0;
    std::vector<size_t> inputDims;
    for (char c : inputSubscripts) {
        if (c != ',') {
            if (dimMap.find(c) == dimMap.end()) {
                dimMap[c] = tensorCount++;
            }
            inputDims.push_back(dimMap[c]);
        }
    }

    std::vector<size_t> resultShape(outputSubscripts.size());
    for (size_t i = 0; i < outputSubscripts.size(); ++i) {
        char outputSubscript = outputSubscripts[i];
        if (dimMap.find(outputSubscript) == dimMap.end()) {
            throw std::invalid_argument("Invalid subscripts string: output subscript not found in input subscripts");
        }
        size_t dimIndex = dimMap.at(outputSubscript);
        resultShape[i] = tensors[dimIndex].getShape()[inputDims[i]];
    }   

    NDArray<T> result(resultShape);
    std::vector<size_t> indices(inputSubscripts.size());
    std::vector<size_t> resultIndices(outputSubscripts.size());

    einsumHelper(subscripts, tensors, indices, resultIndices, result, dimMap, 0, 1.0);
    return result;
}
