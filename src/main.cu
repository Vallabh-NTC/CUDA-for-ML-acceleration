#include "cnn/utils.hpp"
#include <iostream>

int main() {
    // Dummy sizes
    const int N=1, C_=3, H=32, W=32;
    const int K=8, R=3, S=3;
    float *input=nullptr, *filter=nullptr, *output=nullptr;

    cnn::conv_forward(input, filter, output, N,C_,H,W, K,R,S, 1,1,1,1);

    const int M=16, L=8, Kdim=8;
    float *A=nullptr, *B=nullptr, *C=nullptr;
    cnn::gemm_forward(A,B,C,M,L,Kdim);

    std::cout << "Minimal CNN skeleton ran successfully!" << std::endl;
    return 0;
}
