
#include <cufft.h>
#include <cassert>
#ifndef _FLOAT_PRECISION_
#define cufftExecFWD cufftExecD2Z
#define cufftExecBWD cufftExecZ2D
#define cufftPlanFWD CUFFT_D2Z
#define cufftPlanBWD CUFFT_Z2D
#define Real cufftDoubleReal
#define Cmpl cufftDoubleComplex
#else //_FLOAT_PRECISION_
#define cufftExecFWD cufftExecR2C
#define cufftExecBWD cufftExecC2R
#define cufftPlanFWD CUFFT_R2C
#define cufftPlanBWD CUFFT_C2R
#define Real cufftReal
#define Cmpl cufftComplex
#endif//_FLOAT_PRECISION_

void freePlan(cufftHandle& plan) {
  cufftDestroy(plan);
}
void makePlan(cufftHandle& handle, const int mx, const int my, cufftType plan) {
  cufftPlan2d(&handle, mx, my, plan);
}
void freeCuMem(Real* buf) {
  cudaFree(buf);
}
void allocCuMem(Real* & ptr, const size_t size) {
  cudaMalloc((void **)& ptr, size);
}

__global__ void kPeriodic(const int mx, const int my, const int my_hat,
  const Real facX, const Real facY, const Real norm,
  Cmpl*const in_out) {
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

__global__ void kFreespace(const int mx, const int my_hat,
  const Real*const G_hat, Cmpl*const in_out) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= mx) || (j >= my_hat) ) return;
  const int linidx = i * my_hat + j;
  in_out[linidx].x *= G_hat[linidx];
  in_out[linidx].y *= G_hat[linidx];
}

void dPeriodic(const cufftHandle&fwd, const cufftHandle&bwd, const int mx,
 const int my, const Real h, Real*const rhs,Real*const rhs_gpu)
{
  const int my_hat = my/2 +1;
  const Real facX = 2.0*M_PI/(mx*h);
  const Real facY = 2.0*M_PI/(my*h);
  const Real norm = 1./(mx*my);

  cudaMemcpy(rhs_gpu,rhs, 2*mx*my_hat*sizeof(Real), cudaMemcpyHostToDevice);
  cufftExecFWD(fwd, rhs_gpu, (Cmpl*) rhs_gpu);

  dim3 dimB(16, 16);
  assert(mx % dimB.x == 0);
  assert(my % dimB.y == 0);
  dim3 dimG(mx / dimB.x, my_hat / dimB.y + 1);
  kPeriodic <<<dimG,dimB>>> (mx,my,my_hat, facX,facY,norm, (Cmpl*)rhs_gpu);

  cufftExecBWD(bwd, (Cmpl*) rhs_gpu, rhs_gpu);
  cudaMemcpy(rhs,rhs_gpu, 2*mx*my_hat*sizeof(Real), cudaMemcpyDeviceToHost);
}

#include "cstdio"
#include <cuda_runtime.h>

class GpuTimer {
    cudaEvent_t B, E; //cudaEventBlockingSync
  public:
    GpuTimer() { cudaEventCreate(&B); cudaEventCreate(&E); }
    GpuTimer(int flag) { cudaEventCreate(&B, flag); cudaEventCreate(&E, flag); }
    ~GpuTimer() { cudaEventDestroy(B); cudaEventDestroy(E); }
    void start() { cudaEventRecord(B, 0); }
    void stop() { cudaEventRecord(E, 0); }
    float get() {
      float elapsed;
      cudaEventSynchronize(E);
      cudaEventElapsedTime(&elapsed, B, E);
      return elapsed;
    }
};

void dFreespace(const cufftHandle&fwd, const cufftHandle&bwd, const int nx,
  const int ny, Real*const rhs, const Real*const G_hat, Real*const rhs_gpu)
{
  const int mx = 2 * nx - 1, my = 2 * ny - 1;
  const int my_hat = my/2 +1, ny_hat = ny/2 +1;
  //GpuTimer t0, t1, t2, t3, t4;
  //t0.start();
  cudaMemcpy2D(rhs_gpu,2*my_hat*sizeof(Real), rhs,2*ny_hat*sizeof(Real),
    ny*sizeof(Real), nx, cudaMemcpyHostToDevice);
  //t0.stop();
  //t1.start();
  cufftExecFWD(fwd, rhs_gpu, (Cmpl*) rhs_gpu);
  //t1.stop();
  //t2.start();
  dim3 dimB(16, 16);
  assert((mx+1) % dimB.x == 0);
  assert(my_hat % dimB.y == 0);
  dim3 dimG( (mx+1) / dimB.x, my_hat / dimB.y);
  kFreespace <<<dimG,dimB>>> (mx,my_hat, G_hat, (Cmpl*)rhs_gpu);
  //t2.stop();
  //t3.start();
  cufftExecBWD(bwd, (Cmpl*) rhs_gpu, rhs_gpu);
  //t3.stop();
  //t4.start();
  cudaMemcpy2D(rhs,2*ny_hat*sizeof(Real), rhs_gpu,2*my_hat*sizeof(Real),
    ny*sizeof(Real), nx, cudaMemcpyDeviceToHost);
  cudaMemset(rhs_gpu, 0, mx*my_hat * 2*sizeof(Real) );
  //t4.stop();
  //printf("%f %f %f %f %f\n",t0.get(),t1.get(),t2.get(),t3.get(),t4.get());
}

__global__ void kGreen(const int nx, const int ny, const int mx, const int my,
  const int my_hat, const Real fac, const Real h, Real*const in_out) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= mx) || (j >= my) ) return;
  const int linidx = j + 2*my_hat*i;
  const Real xi = i>=nx? mx-i : i;
  const Real yi = j>=ny? my-j : j;
  const Real r = std::sqrt(xi*xi + yi*yi);
  if(r > 0) in_out[linidx] = fac * std::log(h * r);
  // r_eq = h / sqrt(pi)
  // G = 1/4 * r_eq^2 * (2* ln(r_eq) - 1)
  else      in_out[linidx] = fac/2 * (2*std::log(h/std::sqrt(M_PI)) - 1);
}

__global__ void kCopyC2R(const int mx, const int my_hat, const Real norm,
  const Cmpl*const G_hat, Real*const m_kernel) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= mx) || (j >= my_hat) ) return;
  const int linidx = j + my_hat*i;
  m_kernel[linidx] = G_hat[linidx].x * norm;
}

void clearCuMem(Real * buf, const size_t size) { cudaMemset(buf, 0, size ); }

void initGreen(const int nx, const int ny, const Real h, Real*const m_kernel)
{
  const int mx = 2 * nx - 1, my = 2 * ny - 1;
  const int my_hat = my/2 +1;
  Real * tmp;
  cudaMalloc((void **)& tmp, mx * my_hat * sizeof(Cmpl) );
  {
    const Real fac = h * h / ( 2*M_PI );
    dim3 dimB(16, 16);
    assert((mx+1) % dimB.x == 0);
    assert((my+1) % dimB.y == 0);
    dim3 dimG( (mx+1) / dimB.x, (my+1) / dimB.y);
    kGreen<<<dimG, dimB>>> (nx,ny, mx,my, my_hat, fac, h, tmp);
  }
  {
    cufftHandle fwd;
    cufftPlan2d(&fwd, mx, my, cufftPlanFWD);
    cufftExecFWD(fwd, tmp, (Cmpl*) tmp);
    cufftDestroy(fwd);
  }
  {
    const Real norm = 1.0 / (mx * my);
    dim3 dimB(16, 16);
    assert((mx+1) % dimB.x == 0);
    assert(my_hat % dimB.y == 0);
    dim3 dimG( (mx+1) / dimB.x, my_hat / dimB.y);
    kCopyC2R<<<dimG, dimB>>> (mx, my_hat, norm, (Cmpl*)tmp, m_kernel);
  }
  cudaFree(tmp);
}
