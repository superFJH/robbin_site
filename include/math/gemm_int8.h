//
// Created by fjh on 17-11-10.
//

#ifndef MOBILE_DEEP_LEARNING_GEMM_INT8_H
#define MOBILE_DEEP_LEARNING_GEMM_INT8_H
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "gemmlowp.h"
#include "output_stages.h"
#include <fstream>
template <typename tScalar, gemmlowp::MapOrder tOrder>
class MatrixWithStorage {
public:
    MatrixWithStorage(int rows, int cols)
            : storage(rows * cols), matrix_map(storage.data(), rows, cols) {}
    void MakeRandom() {
        static std::mt19937 random_engine;
        std::uniform_real_distribution<float> distribution(-1, 1);
        for (auto& x : storage) {
            x = static_cast<tScalar>(distribution(random_engine));
        }
    }

    gemmlowp::MatrixMap<const tScalar, tOrder> ConstMap() const {
        return gemmlowp::MatrixMap<const tScalar, tOrder>(
                storage.data(), matrix_map.rows(), matrix_map.cols());
    }
    gemmlowp::MatrixMap<tScalar, tOrder> Map() {
        return gemmlowp::MatrixMap<tScalar, tOrder>(
                storage.data(), matrix_map.rows(), matrix_map.cols());
    }
    const std::vector<tScalar>& Storage() const { return storage; }
    std::vector<tScalar>& Storage() { return storage; }

private:
    std::vector<tScalar> storage;
    gemmlowp::MatrixMap<tScalar, tOrder> matrix_map;
};

struct QuantizationParams {
    float scale;
    std::uint8_t zero_point;
};

template <gemmlowp::MapOrder tOrder>
void FindMinMax(const gemmlowp::MatrixMap<float, tOrder>& m, float* min,
                float* max);

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::uint8_t>& src, std::vector<float>* dst);


void Quantize(const QuantizationParams& qparams, const std::vector<float>& src,
              std::vector<std::uint8_t>* dst);


template <gemmlowp::MapOrder tLhsOrder, gemmlowp::MapOrder tRhsOrder,
        gemmlowp::MapOrder tResultOrder>
void FloatMatrixMultiplication(
        const gemmlowp::MatrixMap<const float, tLhsOrder>& lhs,
        const gemmlowp::MatrixMap<const float, tRhsOrder>& rhs,
        gemmlowp::MatrixMap<float, tResultOrder>* result);


void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift);

QuantizationParams ChooseQuantizationParams(float min, float max);
#endif //MOBILE_DEEP_LEARNING_GEMM_INT8_H
