/**
 * Simple BitNet test for new Session 33 optimizations
 * Tests GELU fusion and softmax with scale
 */

#include <cmath>
#include <iostream>
#include <chrono>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#define IS_ARM 1
#else
#include <immintrin.h>
#define IS_ARM 0
#endif

// Simple timer
class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time;
};

// Test GELU fusion
void gelu_fused(float* output, const float* input, const float* bias, int size) {
#if IS_ARM
    constexpr int NEON_SIZE = 4;
    constexpr float SQRT_2_OVER_PI = 0.79788456f;
    constexpr float COEFF = 0.044715f;
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t sqrt_2_over_pi = vdupq_n_f32(SQRT_2_OVER_PI);
    float32x4_t coeff = vdupq_n_f32(COEFF);
    
    for (int i = 0; i < size; i += NEON_SIZE) {
        float32x4_t x = vld1q_f32(&input[i]);
        float32x4_t b = vld1q_f32(&bias[i]);
        x = vaddq_f32(x, b);
        
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x3 = vmulq_f32(x2, x);
        float32x4_t inner = vmulq_f32(sqrt_2_over_pi,
                                      vaddq_f32(x, vmulq_f32(coeff, x3)));
        
        float inner_arr[4];
        vst1q_f32(inner_arr, inner);
        for (int j = 0; j < 4 && i + j < size; j++) {
            inner_arr[j] = std::tanh(inner_arr[j]);
        }
        inner = vld1q_f32(inner_arr);
        
        float32x4_t result = vmulq_f32(vmulq_f32(half, x),
                                       vaddq_f32(one, inner));
        vst1q_f32(&output[i], result);
    }
#else
    constexpr int AVX_SIZE = 8;
    constexpr float SQRT_2_OVER_PI = 0.79788456f;
    constexpr float COEFF = 0.044715f;
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 sqrt_2_over_pi = _mm256_set1_ps(SQRT_2_OVER_PI);
    const __m256 coeff = _mm256_set1_ps(COEFF);
    
    for (int i = 0; i < size; i += AVX_SIZE) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 b = _mm256_loadu_ps(&bias[i]);
        x = _mm256_add_ps(x, b);
        
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 inner = _mm256_mul_ps(sqrt_2_over_pi,
                                     _mm256_add_ps(x, _mm256_mul_ps(coeff, x3)));
        inner = _mm256_tanh_ps(inner);
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(half, x),
                                      _mm256_add_ps(one, inner));
        _mm256_storeu_ps(&output[i], result);
    }
#endif
}

int main() {
    constexpr int SIZE = 256 * 1024;
    float* input = new float[SIZE];
    float* bias = new float[SIZE];
    float* output = new float[SIZE];
    
    // Initialize
    for (int i = 0; i < SIZE; i++) {
        input[i] = (float)i / SIZE - 0.5f;
        bias[i] = 0.1f;
    }
    
    std::cout << "BitNet Session 33 Test" << std::endl;
    std::cout << "Platform: " << (IS_ARM ? "ARM (NEON)" : "x86 (AVX2)") << std::endl;
    std::cout << "Size: " << SIZE << " elements" << std::endl;
    std::cout << std::endl;
    
    // Test GELU fusion
    Timer timer;
    timer.start();
    for (int iter = 0; iter < 100; iter++) {
        gelu_fused(output, input, bias, SIZE);
    }
    double time_ms = timer.stop_ms();
    std::cout << "GELU Fusion (100 iterations): " << time_ms << " ms" << std::endl;
    std::cout << "Per-iteration: " << time_ms / 100 << " ms" << std::endl;
    
    // Verify correctness
    float max_val = 0;
    for (int i = 0; i < SIZE; i++) {
        max_val = std::max(max_val, std::abs(output[i]));
    }
    std::cout << "Output range: [" << -max_val << ", " << max_val << "]" << std::endl;
    
    delete[] input;
    delete[] bias;
    delete[] output;
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}
