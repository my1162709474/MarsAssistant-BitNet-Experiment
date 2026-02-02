// ==================== Session 93: Hyper-Parallel SIMD & Streaming Optimization ====================
// Date: 2026-02-02 08:16
// Target: Additional 5-15% performance through hyper-parallel SIMD and streaming stores
// Focus: Advanced reduction operations, streaming stores, and hardware-accelerated functions

#if defined(__x86_64__) || defined(__i386__)

// ==================== Hyper-Parallel Horizontal Reduction ====================
// 512-way horizontal reduction for maximum throughput

FORCE_INLINE float hyper_reduce_max_ps_avx2(const float* data, int size) {
    constexpr int AVX_SIZE = 8;
    __m256 max_val = _mm256_set1_ps(-INFINITY);
    
    // Process in 256-bit chunks
    for (int i = 0; i <= size - AVX_SIZE; i += AVX_SIZE) {
        max_val = _mm256_max_ps(max_val, _mm256_loadu_ps(data + i));
    }
    
    // Handle remaining elements
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        max_val = _mm256_max_ss(max_val, _mm256_set1_ps(data[i]));
    }
    
    // Horizontal reduction of max_val
    __m256 max_shuffled = _mm256_shuffle_ps(max_val, max_val, 0x4E);  // Swap halves
    max_val = _mm256_max_ps(max_val, max_shuffled);
    max_shuffled = _mm256_shuffle_ps(max_val, max_val, 0xB1);  // Swap within halves
    max_val = _mm256_max_ps(max_val, max_shuffled);
    
    return _mm256_cvtss_f32(max_val);
}

FORCE_INLINE float hyper_reduce_sum_ps_avx2(const float* data, int size) {
    constexpr int AVX_SIZE = 8;
    __m256 sum_val = _mm256_setzero_ps();
    
    // Process in 256-bit chunks
    for (int i = 0; i <= size - AVX_SIZE; i += AVX_SIZE) {
        sum_val = _mm256_add_ps(sum_val, _mm256_loadu_ps(data + i));
    }
    
    // Handle remaining elements
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        sum_val = _mm256_add_ss(sum_val, _mm256_set1_ps(data[i]));
    }
    
    // Horizontal reduction of sum_val
    __m256 sum_shuffled = _mm256_shuffle_ps(sum_val, sum_val, 0x4E);  // Swap halves
    sum_val = _mm256_add_ps(sum_val, sum_shuffled);
    sum_shuffled = _mm256_shuffle_ps(sum_val, sum_val, 0xB1);  // Swap within halves
    sum_val = _mm256_add_ps(sum_val, sum_shuffled);
    
    return _mm256_cvtss_f32(sum_val);
}

// ==================== Streaming Store MatMul ====================
// Non-temporal stores to bypass cache for output matrices

FORCE_INLINE void matmul_streaming_store_avx2(const float* RESTRICT A,
                                               const float* RESTRICT B,
                                               float* RESTRICT C,
                                               int M, int N, int K) {
    constexpr int AVX_SIZE = 8;
    constexpr int UNROLL_N = 4;  // Process 32 floats at a time
    
    // Only use streaming stores for large matrices
    const size_t total_elements = (size_t)M * N;
    const bool use_streaming = total_elements > 1024 * 1024;  // > 1M elements
    
    for (int i = 0; i < M; i++) {
        const float* RESTRICT A_row = A + i * K;
        float* RESTRICT C_row = C + i * N;
        
        for (int j = 0; j < N; j += AVX_SIZE * UNROLL_N) {
            __m256 c[UNROLL_N];
            for (int u = 0; u < UNROLL_N; u++) {
                c[u] = _mm256_setzero_ps();
            }
            
            for (int k = 0; k < K; k++) {
                __m256 a_val = _mm256_set1_ps(A_row[k]);
                
                for (int u = 0; u < UNROLL_N; u++) {
                    int col = j + u * AVX_SIZE;
                    if (col + AVX_SIZE <= N) {
                        __m256 b_vec = _mm256_loadu_ps(B + k * N + col);
                        c[u] = _mm256_fmadd_ps(a_val, b_vec, c[u]);
                    }
                }
            }
            
            // Store with streaming or regular stores
            for (int u = 0; u < UNROLL_N; u++) {
                int col = j + u * AVX_SIZE;
                if (col + AVX_SIZE <= N) {
                    if (use_streaming) {
                        _mm256_stream_ps(C_row + col, c[u]);
                    } else {
                        _mm256_storeu_ps(C_row + col, c[u]);
                    }
                }
            }
        }
    }
    
    // Memory fence for streaming stores
    if (use_streaming) {
        _mm_sfence();
    }
}

// ==================== Hardware-Accelerated Exp Approximation ====================
// Using polynomial approximation for fast exp with AVX2

FORCE_INLINE __m256 fast_exp_ps_avx2(__m256 x) {
    // Constants for exp approximation
    const __m256 exp_high = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_low = _mm256_set1_ps(-88.3762626647949f);
    const __m256 ln2_inv = _mm256_set1_ps(1.4426950408889634f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    
    // Clamp to valid range
    x = _mm256_max_ps(x, exp_low);
    x = _mm256_min_ps(x, exp_high);
    
    // Extract integer and fractional parts
    __m256i x_i = _mm256_cvtps_epi32(_mm256_mul_ps(x, ln2_inv));
    __m256 x_f = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_cvtepi32_ps(x_i), _mm256_set1_ps(0.69314718056f)));
    
    // Polynomial approximation for exp(x_f)
    __m256 x_f2 = _mm256_mul_ps(x_f, x_f);
    __m256 x_f4 = _mm256_mul_ps(x_f2, x_f2);
    
    // exp(x_f) ≈ 1 + x + x²/2! + x³/3! + x⁴/4!
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.9999999999999999f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666666666666666f);
    const __m256 c4 = _mm256_set1_ps(0.041666666666666664f);
    
    __m256 result = _mm256_add_ps(c0, _mm256_mul_ps(x_f, c1));
    result = _mm256_add_ps(result, _mm256_mul_ps(x_f2, c2));
    result = _mm256_add_ps(result, _mm256_mul_ps(x_f4, _mm256_mul_ps(x_f2, c3)));
    result = _mm256_add_ps(result, _mm256_mul_ps(x_f4, _mm256_mul_ps(x_f4, c4)));
    
    // Multiply by 2^x_i
    __m256i two_pow_x_i = _mm256_add_epi32(x_i, _mm256_set1_epi32(127));
    __m256i mantissa_and_exp = _mm256_slli_epi32(two_pow_x_i, 23);
    
    return _mm256_mul_ps(result, _mm256_castsi256_ps(mantissa_and_exp));
}

// ==================== Ultra-Fast Softmax with Streaming ====================
// Optimized softmax with fast exp and streaming-friendly access

FORCE_INLINE void softmax_ultra_fast_avx2(float* data, int size) {
    // Find max value and compute sum
    float max_val = hyper_reduce_max_ps_avx2(data, size);
    __m256 max_vec = _mm256_set1_ps(max_val);
    
    // Compute exp and sum
    __m256 sum = _mm256_setzero_ps();
    
    constexpr int AVX_SIZE = 8;
    for (int i = 0; i <= size - AVX_SIZE; i += AVX_SIZE) {
        __m256 x = _mm256_loadu_ps(data + i);
        x = _mm256_sub_ps(x, max_vec);
        __m256 exp_x = fast_exp_ps_avx2(x);
        sum = _mm256_add_ps(sum, exp_x);
        _mm256_storeu_ps(data + i, exp_x);
    }
    
    // Handle remaining elements
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        float x = data[i] - max_val;
        float exp_x = std::exp(x);
        data[i] = exp_x;
        sum = _mm256_add_ss(sum, _mm256_set1_ps(exp_x));
    }
    
    // Horizontal reduction
    float sum_val = hyper_reduce_sum_ps_avx2_ps(sum);
    
    // Normalize
    __m256 inv_sum = _mm256_set1_ps(1.0f / sum_val);
    
    for (int i = 0; i <= size - AVX_SIZE; i += AVX_SIZE) {
        __m256 x = _mm256_loadu_ps(data + i);
        x = _mm256_mul_ps(x, inv_sum);
        _mm256_storeu_ps(data + i, x);
    }
    
    for (int i = size - (size % AVX_SIZE); i < size; i++) {
        data[i] /= sum_val;
    }
}

// ==================== Batch MatMul with Dynamic Batching ====================
// Dynamic batch sizing based on matrix dimensions

FORCE_INLINE void matmul_dynamic_batch_avx2(const float* A, const float* B, float* C,
                                             int M, int N, int K, int batch_size) {
    // Dynamic batch size based on cache size
    constexpr size_t L1_CACHE = 32 * 1024;
    constexpr size_t L2_CACHE = 256 * 1024;
    
    size_t matrix_size = (size_t)M * N * sizeof(float);
    size_t a_size = (size_t)M * K * sizeof(float);
    size_t b_size = (size_t)K * N * sizeof(float);
    
    // Select optimal batch count
    int optimal_batches;
    if (matrix_size > L2_CACHE) {
        optimal_batches = std::min(batch_size, 2);  // Process 2 at a time
    } else if (matrix_size > L1_CACHE) {
        optimal_batches = std::min(batch_size, 4);  // Process 4 at a time
    } else {
        optimal_batches = std::min(batch_size, 8);  // Process 8 at a time
    }
    
    // Process in batches
    for (int b = 0; b < batch_size; b += optimal_batches) {
        int current_batch = std::min(optimal_batches, batch_size - b);
        
        for (int i = 0; i < M; i++) {
            for (int batch_idx = 0; batch_idx < current_batch; batch_idx++) {
                const float* A_batch = A + (b + batch_idx) * M * K + i * K;
                float* C_batch = C + (b + batch_idx) * M * N + i * N;
                
                // Simple matmul for each batch element
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A_batch[k] * B[k * N + j];
                    }
                    C_batch[j] = sum;
                }
            }
        }
    }
}

// ==================== Advanced Vectorized GELU ====================
// Optimized GELU with polynomial approximation

FORCE_INLINE __m256 fast_gelu_ps_avx2(__m256 x) {
    // Constants for GELU approximation
    const __m256 sqrt_2_over_pi = _mm256_set1_ps(0.7978845608028654f);
    const __m256 coeff = _mm256_set1_ps(0.044715f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    
    // GELU ≈ 0.5 * x * (1 + tanh(0.797885 * x * (1 + 0.044715 * x²)))
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 inner = _mm256_mul_ps(x, _mm256_add_ps(one, _mm256_mul_ps(coeff, x2)));
    __m256 tanh_inner = fast_tanh_ps_avx2(_mm256_mul_ps(sqrt_2_over_pi, inner));
    
    return _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, tanh_inner));
}

// ==================== Cache-Optimized Attention ====================
// Attention with cache-friendly access patterns

FORCE_INLINE void attention_cache_optimized_avx2(const float* Q, const float* K, const float* V,
                                                  float* output, int batch_size, int num_heads,
                                                  int seq_len, int head_dim) {
    const int total_heads = batch_size * num_heads;
    const int H = head_dim;
    const int N = seq_len;
    
    for (int h = 0; h < total_heads; h++) {
        const float* Q_head = Q + h * H;
        const float* K_head = K + h * H;
        const float* V_head = V + h * H;
        float* O_head = output + h * H;
        
        // Q @ K^T (attention scores)
        float* scores = (float*)tl_alloc(N * N * sizeof(float));
        
        // Blocked computation for cache efficiency
        constexpr int BLOCK_SIZE = 64;
        
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                // Process block
                for (int ii = i; ii < std::min(i + BLOCK_SIZE, N); ii++) {
                    for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); jj++) {
                        float sum = 0.0f;
                        for (int d = 0; d < H; d++) {
                            sum += Q_head[ii * H + d] * K_head[jj * H + d];
                        }
                        scores[ii * N + jj] = sum / std::sqrt(H);
                    }
                }
            }
        }
        
        // Softmax
        softmax_ultra_fast_avx2(scores, N * N);
        
        // Softmax @ V
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < H; d++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    sum += scores[i * N + j] * V_head[j * H + d];
                }
                O_head[i * H + d] = sum;
            }
        }
        
        tl_free(scores);
    }
}

#endif  // x86_64

// ==================== ARM NEON Optimizations ====================
#if defined(__aarch64__) || defined(__arm64__)

// ==================== NEON Streaming Store ====================

FORCE_INLINE void matmul_streaming_store_neon(const float* RESTRICT A,
                                               const float* RESTRICT B,
                                               float* RESTRICT C,
                                               int M, int N, int K) {
    constexpr int NEON_SIZE = 4;
    constexpr int UNROLL_N = 4;
    
    const size_t total_elements = (size_t)M * N;
    const bool use_streaming = total_elements > 1024 * 1024;
    
    for (int i = 0; i < M; i++) {
        const float* RESTRICT A_row = A + i * K;
        float* RESTRICT C_row = C + i * N;
        
        for (int j = 0; j < N; j += NEON_SIZE * UNROLL_N) {
            float32x4_t c[UNROLL_N];
            for (int u = 0; u < UNROLL_N; u++) {
                c[u] = vdupq_n_f32(0.0f);
            }
            
            for (int k = 0; k < K; k++) {
                float32x4_t a_val = vdupq_n_f32(A_row[k]);
                
                for (int u = 0; u < UNROLL_N; u++) {
                    int col = j + u * NEON_SIZE;
                    if (col + NEON_SIZE <= N) {
                        float32x4_t b_vec = vld1q_f32(B + k * N + col);
                        c[u] = vfmaq_f32(c[u], a_val, b_vec);
                    }
                }
            }
            
            for (int u = 0; u < UNROLL_N; u++) {
                int col = j + u * NEON_SIZE;
                if (col + NEON_SIZE <= N) {
                    vst1q_f32(C_row + col, c[u]);
                }
            }
        }
    }
}

// ==================== NEON Fast Softmax ====================

FORCE_INLINE void softmax_fast_neon(float* data, int size) {
    // Find max
    float32x4_t max_val = vdupq_n_f32(-INFINITY);
    int i = 0;
    
    for (; i <= size - 4; i += 4) {
        max_val = vmaxq_f32(max_val, vld1q_f32(data + i));
    }
    
    // Horizontal max reduction
    float32x2_t max_pair = vpmax_f32(vget_low_f32(max_val), vget_high_f32(max_val));
    float max_scalar = vget_lane_f32(vpmax_f32(max_pair, max_pair), 0);
    
    for (; i < size; i++) {
        max_scalar = std::max(max_scalar, data[i]);
    }
    
    max_val = vdupq_n_f32(max_scalar);
    
    // Compute exp and sum
    float32x4_t sum = vdupq_n_f32(0.0f);
    i = 0;
    
    for (; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(data + i);
        x = vsubq_f32(x, max_val);
        // Fast exp approximation
        x = fast_exp_neon(x);
        sum = vaddq_f32(sum, x);
        vst1q_f32(data + i, x);
    }
    
    // Horizontal sum reduction
    float32x2_t sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float sum_scalar = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
    
    for (; i < size; i++) {
        float x = data[i] - max_scalar;
        x = std::exp(x);
        data[i] = x;
        sum_scalar += x;
    }
    
    // Normalize
    float32x4_t inv_sum = vdupq_n_f32(1.0f / sum_scalar);
    i = 0;
    
    for (; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(data + i);
        x = vmulq_f32(x, inv_sum);
        vst1q_f32(data + i, x);
    }
    
    for (; i < size; i++) {
        data[i] /= sum_scalar;
    }
}

#endif  // ARM64

// ==================== Unified Interface ====================

// Select optimal implementation based on compile-time flags
FORCE_INLINE void matmul_optimized(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
#if defined(__x86_64__) || defined(__i386__)
    matmul_streaming_store_avx2(A, B, C, M, N, K);
#elif defined(__aarch64__) || defined(__arm64__)
    matmul_streaming_store_neon(A, B, C, M, N, K);
#else
    // Fallback to basic implementation
    matmul_basic(A, B, C, M, N, K);
#endif
}

FORCE_INLINE void softmax_optimized(float* data, int size) {
#if defined(__x86_64__) || defined(__i386__)
    softmax_ultra_fast_avx2(data, size);
#elif defined(__aarch64__) || defined(__arm64__)
    softmax_fast_neon(data, size);
#else
    softmax_basic(data, size);
#endif
}

#endif  // SESSION 93
