#include <complex>
#include <vector>
#include <cmath>
#include <cstdint>

// Extern "C" flattens the signature so Python's ctypes can load it easily.
extern "C" {

    /**
     * @brief Computes the permanent of a complex square matrix using the BBFG algorithm.
     * * @param A_real Pointer to the flattened real values of the NxN matrix.
     * @param A_imag Pointer to the flattened imaginary values of the NxN matrix.
     * @param N The dimension of the matrix.
     * @param out_real Pointer to store the real part of the resulting permanent.
     * @param out_imag Pointer to store the imaginary part of the resulting permanent.
     */
    void bbfg_permanent(const double* A_real, const double* A_imag, int N, double* out_real, double* out_imag) {
        if (N == 0) {
            *out_real = 1.0;
            *out_imag = 0.0;
            return;
        }

        // Initialize row sums and delta array.
        // delta tracks the sign flips (+1 or -1) for each row.
        std::vector<std::complex<double>> row_sums(N, std::complex<double>(0.0, 0.0));
        std::vector<int> delta(N, 1);

        // Step 1: Initialize row_sums with the sum of all columns (since all delta_i = 1 initially)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int idx = i * N + j;
                row_sums[j] += std::complex<double>(A_real[idx], A_imag[idx]);
            }
        }

        std::complex<double> total(0.0, 0.0);
        std::complex<double> prod(1.0, 0.0);
        
        // Calculate the initial product of the row sums
        for (int j = 0; j < N; ++j) {
            prod *= row_sums[j];
        }
        total += prod;

        int sign = 1;
        // The outer loop runs 2^{N-1} - 1 times.
        uint64_t num_loops = (1ULL << (N - 1)) - 1;

        // Step 2: Iterate through the Gray code sequence
        for (uint64_t k = 1; k <= num_loops; ++k) {
            // Find the index of the bit that flips in the Gray code.
            // __builtin_ctzll returns the number of trailing 0-bits (lowest set bit).
            // We add 1 because delta_0 is permanently fixed to 1 in Glynn's formula.
            int p = __builtin_ctzll(k) + 1;

            // Flip the sign of the modified row and the global product sign
            delta[p] = -delta[p];
            sign = -sign;

            prod = std::complex<double>(1.0, 0.0);
            
            // Update the running row sums
            for (int j = 0; j < N; ++j) {
                int idx = p * N + j;
                std::complex<double> A_pj(A_real[idx], A_imag[idx]);
                
                // Because delta[p] flipped, the difference added to the sum 
                // is exactly 2 * the new delta[p] * the matrix element.
                row_sums[j] += 2.0 * delta[p] * A_pj;
                prod *= row_sums[j];
            }

            // Accumulate into the total permanent
            if (sign == 1) {
                total += prod;
            } else {
                total -= prod;
            }
        }

        // Step 3: Divide by the normalization factor 2^{N-1}
        double norm = std::pow(2.0, N - 1);
        *out_real = total.real() / norm;
        *out_imag = total.imag() / norm;
    }
}