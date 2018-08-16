#include "PoissonSolverCuda.h"

__device__ static inline void solve(const int mx,const int my, const int my_hat,
  const cufftValT facX, const cufftValT facY, const cufftValT norm,
  ccufftCmpT*const in_out)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= mx) || (j >= my_hat) ) return;
  const int linidx = i * my_hat + j;
  const int kx = (i <= mx/2) ? i : -(mx-i);
  const int ky = (j <= my/2) ? j : -(my-j);
  const Real rkx = kx*facX, rky = ky*facY;
  const Real kinv = (kx==0 && ky==0) ? 0 : -(Real)1/(rkx*rkx+rky*rky);
  in_out[linidx].x *= norm * kinv;
  in_out[linidx].y *= norm * kinv;
}

void PoissonSolverCuda::solve() const
{
  cudaMemcpy(rhs, rhs_gpu, 2*mx*my_hat*sizeof(Real), cudaMemcpyHostToDevice);
  cufftExecFWD(fwd, rhs_gpu, (cufftCmpT*) rhs_gpu);

  dim3 dimBlock(16, 16);
  dim3 dimGrid(N / dimBlock.x, (my/2) / dimBlock.y + 1);
  solve <<<dimGrid, dimBlock>>> (mx, my, my_hat, facX, facY, norm, rhs_gpu);

  cufftExecBWD(bwd, (cufftCmpT*) rhs_gpu, rhs_gpu);
  cudaMemcpy(rhs_gpu, rhs, 2*mx*my_hat*sizeof(Real), cudaMemcpyDeviceToHost);

  _fftw2cub();
}

PoissonSolverCuda::PoissonSolverCuda(FluidGrid& _grid, const bool bFrespace):
grid(_grid), mx(bFrespace? 2 * nx - 1 : nx), my(bFrespace? 2 * ny - 1 : ny)
{
  cufftPlan2d(&fwd, mx, my, CUFFT_R2C);
  cufftPlan2d(&bwd, mx, my, CUFFT_C2R);
  assert(2*sizeof(Real) == sizeof(ccufftCmpT));
  rhs = (Real*) malloc(mx * my_hat * 2 * sizeof(Real) );
  cudaMalloc((void **)& gpu_rhs, mx * my_hat * sizeof(ccufftCmpT) );
}


//__global__ void complex2RealScaled(float2 * __restrict__ d_r,
//  float * __restrict__ d_result, const int M, const int N, float scale)
//{
//    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
//    if ((tidx >= M) || (tidy >= N)) return;
//    d_result[tidy * M + tidx] = scale * (d_r[tidy * M + tidx].x - d_r[0].x);
//}
