#include <sys/time.h>

// #include </root/Kokkos/kokkos/core/src/Kokkos_Core.hpp>
// #include </root/Kokkos/kokkos-install/include/Kokkos_Core.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>


class matrix_operation {
 public:
  matrix_operation() {
    // std::cout << "matrix_operation()" << std::endl;
  }
  ~matrix_operation() {
    // std::cout << "~matrix_operation()" << std::endl;
  }
  void matrix_multiply(double *A, double *B, double *C, int n) {
    // std::cout << "matrix_multiply()" << std::endl;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(int i) {
          for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
              C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
          }
        });
  }
  void matrix_multiply_parallel(double *A, double *B, double *C, int n) {
    // std::cout << "matrix_multiply_parallel()" << std::endl;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, n),
        KOKKOS_LAMBDA(int i) {
          for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
              C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
          }
        });
  }
  void matrix_multiply_serial(double *A, double *B, double *C, int n) {
    // std::cout << "matrix_multiply_serial()" << std::endl;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        C[i * n + j] = 0;
        for (int k = 0; k < n; k++) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
  }
  void matrix_multiply_serial_parallel(double *A, double *B, double *C, int n) {
    // std::cout << "matrix_multiply_serial_parallel()" << std::endl;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        C[i * n + j] = 0;
        for (int k = 0; k < n; k++) {
          C[i * n + j] += A[i * n + k] * B[k * n + j];
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
  int N = 65536;      // number of rows 2^12
  int M = 1024;       // number of columns 2^10
  int nrepeat = 100;  // number of repeats of the test

  Kokkos::initialize(argc, argv);
  {
    {
      /******************************************/
      /******************************************/
      typedef Kokkos::View<double *> ViewVectorType;
      typedef Kokkos::View<double **> ViewMatrixType;
      ViewVectorType y("y", N);
      ViewVectorType x("x", M);
      ViewMatrixType A("A", N, M);

      // Initialize y vector on host.
      for (i = 0; i < N; ++i) {
        y(i) = 1;
      }

      // Initialize x vector on host.
      for (int i = 0; i < M; ++i) {
        x(i) = 1;
      }

      // Initialize A matrix on host, note 2D indexing.
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
          A(j, i) = 1;
        }
      }

      // Timer products.
      Kokkos::Timer timer1;

      for (int repeat = 0; repeat < nrepeat; repeat++) {
        // Application: <y,Ax> = y^T*A*x
        double result = 0;

        Kokkos::parallel_reduce(
            "yAx", N, KOKKOS_LAMBDA(int j, double &update) {
              double temp2 = 0;

              for (int i = 0; i < M; ++i) {
                temp2 += A(j, i) * x(i);
              }

              update += y(j) * temp2;
            },
            result);
      }

      // Calculate time.
      double time1 = timer1.seconds();
      printf("Compute( y^T*A*x ),  A( %d ) x( %d ) y( %d ) nrepeat( %d ) time( %g s )\n", M * N, M, N, nrepeat, time1);
    }
  }
  Kokkos::finalize();

  return 0;
}
