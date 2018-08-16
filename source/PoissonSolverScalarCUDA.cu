
#include <cufft.h>

#ifndef _FLOAT_PRECISION_
#define cufftExecFWD cufftExecR2C
#define cufftExecBWD cufftExecC2R
#define cufftPlanFWD CUFFT_R2C
#define cufftPlanBWD CUFFT_C2R
#define cufftValT cufftReal
#define cufftCmpT cufftComplex
#else //_FLOAT_PRECISION_
#define cufftExecFWD cufftExecD2Z
#define cufftExecBWD cufftExecZ2D
#define cufftPlanFWD CUFFT_D2Z
#define cufftPlanBWD CUFFT_Z2D
#define cufftValT cufftDoubleReal
#define cufftCmpT cufftDoubleComplex
#endif//_FLOAT_PRECISION_

void freePlan(cufftHandle& plan) {
  cufftDestroy(plan);
}
void makePlan(cufftHandle& handle, const int mx, const int my, cufftType plan) {
  cufftPlan2d(&handle, mx, my, plan);
}
void freeCuMem(cufftValT* buf) {
  cudaFree(buf);
}
void allocCuMem(cufftValT* & ptr, const size_t size) {
  cudaMalloc((void **)& ptr, size);
}

__global__ void solve(const int mx, const int my, const int my_hat,
  const cufftValT facX, const cufftValT facY, const cufftValT norm,
  cufftCmpT*const in_out)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= mx) || (j >= my_hat) ) return;
  const int linidx = i * my_hat + j;
  const int kx = (i <= mx/2) ? i : -(mx-i);
  const int ky = (j <= my/2) ? j : -(my-j);
  const cufftValT rkx = kx*facX, rky = ky*facY;
  const cufftValT kinv = (kx==0 && ky==0) ? 0 : -(cufftValT)1/(rkx*rkx+rky*rky);
  in_out[linidx].x *= norm * kinv;
  in_out[linidx].y *= norm * kinv;
}

void cuSolve(const cufftHandle&fwd, const cufftHandle&bwd, const int mx,
  const int my, const cufftValT h, cufftValT*const rhs, cufftValT*const rhs_gpu)
{
  const int my_hat = my/2 +1;
  const cufftValT facX = 2.0*M_PI/(mx*h);
  const cufftValT facY = 2.0*M_PI/(my*h);
  const cufftValT norm = 1./(mx*my);

  cudaMemcpy(rhs,rhs_gpu, 2*mx*my_hat*sizeof(cufftValT),cudaMemcpyHostToDevice);
  cufftExecFWD(fwd, rhs_gpu, (cufftCmpT*) rhs_gpu);

  dim3 dimBlock(16, 16);
  dim3 dimGrid(mx / dimBlock.x, (my/2) / dimBlock.y + 1);
  solve<<<dimGrid,dimBlock>>> (mx,my,my_hat,facX,facY,norm,(cufftCmpT*)rhs_gpu);

  cufftExecBWD(bwd, (cufftCmpT*) rhs_gpu, rhs_gpu);
  cudaMemcpy(rhs_gpu,rhs, 2*mx*my_hat*sizeof(cufftValT),cudaMemcpyDeviceToHost);
}


//__global__ void complex2RealScaled(float2 * __restrict__ d_r,
//  float * __restrict__ d_result, const int M, const int N, float scale)
//{
//    const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//    const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
//    if ((tidx >= M) || (tidy >= N)) return;
//    d_result[tidy * M + tidx] = scale * (d_r[tidy * M + tidx].x - d_r[0].x);
//}
