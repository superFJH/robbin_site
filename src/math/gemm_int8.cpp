//
// Created by fjh on 17-11-10.
//
#include "math/gemm_int8.h"
// Find the min and max value in a float matrix.
template <gemmlowp::MapOrder tOrder>
void FindMinMax(const gemmlowp::MatrixMap<float, tOrder>& m, float* min,
                float* max) {
    *min = *max = m(0, 0);
    for (int i = 0; i < m.rows(); i++) {
        for (int j = 0; j < m.cols(); j++) {
            const float val = m(i, j);
            *min = std::min(*min, val);
            *max = std::max(*max, val);
        }
    }
}

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.


// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
QuantizationParams ChooseQuantizationParams(float min, float max) {
    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min = std::min(min, 0.f);
    max = std::max(max, 0.f);

    // the min and max quantized values, as floating-point values
    const float qmin = 0;
    const float qmax = 255;

    // First determine the scale.
    const double scale = (max - min) / (qmax - qmin);

    // Zero-point computation.
    // First the initial floating-point computation. The zero-point can be
    // determined from solving an affine equation for any known pair
    // (real value, corresponding quantized value).
    // We know two such pairs: (rmin, qmin) and (rmax, qmax).
    // Let's use the first one here.
    const double initial_zero_point = qmin - min / scale;

    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the requirement
    // to be able to represent the real value "0" exactly as a quantized value,
    // which is required in multiple places, for example in Im2col with SAME
    // padding).
    std::uint8_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
                static_cast<std::uint8_t>(std::round(initial_zero_point));
    }

    QuantizationParams result;
    result.scale = scale;
    result.zero_point = nudged_zero_point;
    return result;
}

template <gemmlowp::MapOrder tLhsOrder, gemmlowp::MapOrder tRhsOrder,
        gemmlowp::MapOrder tResultOrder>
void FloatMatrixMultiplication(
        const gemmlowp::MatrixMap<const float, tLhsOrder>& lhs,
        const gemmlowp::MatrixMap<const float, tRhsOrder>& rhs,
        gemmlowp::MatrixMap<float, tResultOrder>* result) {
    assert(lhs.cols() == rhs.rows());
    assert(lhs.rows() == result->rows());
    assert(rhs.cols() == result->cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int k = 0; k < rhs.cols(); k++) {
            (*result)(i, k) = 0;
            for (int j = 0; j < lhs.cols(); j++) {
                (*result)(i, k) += lhs(i, j) * rhs(j, k);
            }
        }
    }
}

void Quantize(const QuantizationParams& qparams, const std::vector<float>& src,
              std::vector<std::uint8_t>* dst) {
    assert(src.size() == dst->size());
    for (std::size_t i = 0; i < src.size(); i++) {
        const float real_val = src[i];
        const float transformed_val = qparams.zero_point + real_val / qparams.scale;
        const float clamped_val = std::max(0.f, std::min(255.f, transformed_val));
        (*dst)[i] = static_cast<std::uint8_t>(std::round(clamped_val));
    }
}

void Dequantize(const QuantizationParams& qparams,
                const std::vector<std::uint8_t>& src, std::vector<float>* dst) {
    assert(src.size() == dst->size());
    for (std::size_t i = 0; i < src.size(); i++) {
        const std::uint8_t quantized_val = src[i];
        (*dst)[i] = qparams.scale * (quantized_val - qparams.zero_point);
    }
}

void QuantizeMultiplierSmallerThanOne(float real_multiplier,
                                      std::int32_t* quantized_multiplier,
                                      int* right_shift) {
    assert(real_multiplier > 0.f);
    assert(real_multiplier < 1.f);
    int s = 0;
    // We want to bring the real multiplier into the interval [1/2, 1).
    // We can do so by multiplying it by two, and recording how many times
    // we multiplied by two so that we can compensate that by a right
    // shift by the same amount.
    while (real_multiplier < 0.5f) {
        real_multiplier *= 2.0f;
        s++;
    }
    // Now that the real multiplier is in [1/2, 1), we convert it
    // into a fixed-point number.
    std::int64_t q =
            static_cast<std::int64_t>(std::round(real_multiplier * (1ll << 31)));
    assert(q <= (1ll << 31));
    // Handle the special case when the real multiplier was so close to 1
    // that its fixed-point approximation was undistinguishable from 1.
    // We handle this by dividing it by two, and remembering to decrement
    // the right shift amount.
    if (q == (1ll << 31)) {
        q /= 2;
        s--;
    }
    assert(s >= 0);
    assert(q <= std::numeric_limits<std::int32_t>::max());
    *quantized_multiplier = static_cast<std::int32_t>(q);
    *right_shift = s;
}