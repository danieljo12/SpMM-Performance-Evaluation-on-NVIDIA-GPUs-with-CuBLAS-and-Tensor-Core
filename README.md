```markdown
# SpMM Test with cuBLAS and Tensor Cores

This project is a CUDA-based C++ program that uses the cuBLAS library to perform Sparse Matrix-Matrix Multiplication (SpMM) on the GPU. It leverages FP16 (half-precision floating-point) matrices and GPU tensor cores to optimize matrix operations and benchmark performance.

## Requirements

- CUDA Toolkit
- cuBLAS library
- A compatible NVIDIA GPU that supports FP16 and tensor cores

## Building the Project

To compile the code, make sure you have CUDA and cuBLAS installed, then use the following command:

```bash
nvcc -o spmm_test spmm_test.cu -lcublas
```

## Usage

Run the compiled program with the following command-line arguments:

```bash
./spmm_test M K N [Sparsity]
```

### Parameters

- `M`: Number of rows in Matrix A and the output matrix.
- `K`: Number of columns in Matrix A and rows in Matrix B.
- `N`: Number of columns in Matrix B and the output matrix.
- `[Sparsity]` (optional): Percentage of sparsity in Matrix A (defaults to 0 if not provided).

### Example

```bash
./spmm_test 1024 512 1024 20
```

This example runs the program with a 1024x512 matrix A, a 512x1024 matrix B, and a 20% sparsity level on Matrix A.

## Code Breakdown

1. **Memory Allocation**: Allocates memory for matrices A and B on both the CPU and GPU.
2. **Data Initialization**: Initializes matrices with optional sparsity for Matrix A.
3. **Data Transfer to GPU**: Transfers matrices from host memory (CPU) to device memory (GPU).
4. **Matrix Multiplication with cuBLAS**:
   - **Without Tensor Cores**: Executes multiplication using cuBLAS in pedantic math mode.
   - **With Tensor Cores**: Executes multiplication using cuBLAS in default math mode to leverage tensor cores.
5. **Benchmarking**: Measures execution time for each math mode and calculates performance in TFLOPS.
6. **Result Output**: Displays matrix dimensions, sparsity level, and performance results.
7. **Cleanup**: Frees all allocated memory on both CPU and GPU.

## Output

The program outputs:

- Matrix dimensions and sparsity level.
- Execution time and TFLOPS performance metrics.

### Sample Output

```plaintext
******************************************Problem Size******************************************
M: 1024 N: 1024 K: 512 Pruning Rate: 20
******************************************Performance*******************************************
CuBlas_TC: Time: 5.12 ms, TFLOPS: 4.56
```

## Performance Measurement

The program benchmarks two scenarios:

- **Without Tensor Cores** (pedantic math mode)
- **With Tensor Cores** (default math mode)

Execution time and TFLOPS are calculated for each case, allowing performance comparison of tensor core usage.

## Error Handling

The program includes error checking for memory allocations and cuBLAS API calls, ensuring issues are identified during execution.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

This `README.md` provides an overview, installation instructions, usage guide, code breakdown, example output, and performance details for the project.
