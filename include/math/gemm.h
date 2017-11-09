/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/
#ifndef MDL_GEMM_H
#define MDL_GEMM_H
#include "commons/commons.h"

namespace mdl {
    constexpr const int MC = 384;

    constexpr const int KC = 384;

    constexpr const int NC = 4096;

    constexpr const int MR = 4;

    constexpr const int NR = 4;

    struct Gemmer {
        static vector<Gemmer *> gemmers;

        Mtype A_[MC * KC] __attribute__ ((aligned (32)));

        Mtype B_[KC * NC] __attribute__ ((aligned (32)));

        Mtype C_[MR * NR] __attribute__ ((aligned (32)));

        Mtype AB_[MR * NR] __attribute__ ((aligned (32)));

        void pack_MRxk(int k, const Mtype *A, int incRowA, int incColA, Mtype *buffer);

        void pack_A(int mc, int kc, const Mtype *A, int incRowA, int incColA, Mtype *buffer);

        void pack_kxNR(int k, const Mtype *B, int incRowB, int incColB, Mtype *buffer);

        void pack_B(int kc, int nc, const Mtype *B, int incRowB, int incColB, Mtype *buffer);

        void dgemm_micro_kernel(int kc, Mtype alpha, const Mtype *A, const Mtype *B, Mtype beta, Mtype *C, int incRowC, int incColC);

        void dgeaxpy(int m, int n, Mtype alpha, const Mtype *X, int incRowX, int incColX, Mtype *Y, int incRowY, int incColY);

        void dgescal(int m, int n, Mtype alpha, Mtype *X, int incRowX, int incColX);

        void dgemm_macro_kernel(int mc, int nc, int kc, Mtype alpha, Mtype beta, Mtype *C, int incRowC, int incColC);

        void dgemm_nn(int m, int n, int k, Mtype alpha, const Mtype *A, int incRowA, int incColA, const Mtype *B, int incRowB, int incColB, Mtype beta, Mtype *C, int incRowC, int incColC);

        void sgemm(int m, int n, int k, const Mtype *A, const Mtype *B, Mtype *C);

        void sgemm(int m, int n, int k, const Mtype *A, const Mtype *B, Mtype *C, Mtype alpha, Mtype beta);
    };
};
#endif
