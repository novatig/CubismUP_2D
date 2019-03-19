//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


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
void makePlan(cufftHandle& handle, const int MY, const int MX, cufftType plan) {
  cufftPlan2d(&handle, MY, MX, plan);
}
void freeCuMem(Real* buf) {
  cudaFree(buf);
}
void allocCuMem(Real* & ptr, const size_t size) {
  cudaMalloc((void **)& ptr, size);
}

__global__ void kPeriodic(const int MY, const int MX, const int MX_hat,
  const Real facX, const Real facY, const Real norm, Cmpl*const in_out)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (j >= MY) || (i >= MX_hat) ) return;
  const int kx = (i<=MX/2) ? i : -(MX-i);
  const int ky = (j<=MY/2) ? j : -(MY-j);
  const Real rkx = kx*facX, rky = ky*facY;
  const Real kinv = (kx==0 && ky==0) ? 0 : -(Real)1/(rkx*rkx + rky*rky);
  in_out[j * MX_hat + i].x *= norm * kinv;
  in_out[j * MX_hat + i].y *= norm * kinv;
}

__global__ void kFreespace(const int MY, const int MX_hat,
  const Real*const G_hat, Cmpl*const in_out)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (j >= MY) || (i >= MX_hat) ) return;
  in_out[j * MX_hat + i].x *= G_hat[j * MX_hat + i];
  in_out[j * MX_hat + i].y *= G_hat[j * MX_hat + i];
}

void dPeriodic(const cufftHandle&fwd, const cufftHandle&bwd, const int MY,
 const int MX, const Real h, Real*const rhs, Real*const rhs_gpu)
{
  const int MX_hat = MX/2 +1;
  const Real facX = 2*M_PI/MX, facY = 2*M_PI/MY, norm = 1.0/(MX*MY);

  cudaMemcpy(rhs_gpu, rhs, 2*MY*MX_hat*sizeof(Real), cudaMemcpyHostToDevice);
  cufftExecFWD(fwd, rhs_gpu, (Cmpl*) rhs_gpu);

  dim3 dimB(16, 16), dimG(MX_hat / dimB.x + 1, MY / dimB.y);
  assert((MX % dimB.x == 0) && (MY % dimB.y == 0));
  kPeriodic <<<dimG,dimB>>> (MY, MX, MX_hat, facX,facY,norm, (Cmpl*)rhs_gpu);

  cufftExecBWD(bwd, (Cmpl*) rhs_gpu, rhs_gpu);
  cudaMemcpy(rhs,rhs_gpu, 2*MY*MX_hat*sizeof(Real), cudaMemcpyDeviceToHost);
}

#include "cstdio"
#include <cuda_runtime.h>

class GpuTimer
{
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

void dFreespace(const cufftHandle&fwd, const cufftHandle&bwd, const int NX,
  const int NY, Real*const rhs, const Real*const G_hat, Real*const rhs_gpu)
{
  const int MX = 2 * NX - 1, MY = 2 * NY - 1, MX_hat = MX/2 +1;
  //GpuTimer t0, t1, t2, t3, t4;
  //t0.start();
  cudaMemcpy2D(rhs_gpu, 2*MX_hat*sizeof(Real), rhs, NX*sizeof(Real),
    NX*sizeof(Real), NY, cudaMemcpyHostToDevice);
  //t0.stop();
  //t1.start();
  cufftExecFWD(fwd, rhs_gpu, (Cmpl*) rhs_gpu);
  //t1.stop();
  //t2.start();
  dim3 dimB(16, 16), dimG( MX_hat / dimB.x, (MY+1) / dimB.y);
  assert( ((MY+1) % dimB.y) == 0 && (MX_hat % dimB.x) == 0 );
  kFreespace <<<dimG,dimB>>> (MY, MX_hat, G_hat, (Cmpl*)rhs_gpu);
  //t2.stop();
  //t3.start();
  cufftExecBWD(bwd, (Cmpl*) rhs_gpu, rhs_gpu);
  //t3.stop();
  //t4.start();
  cudaMemcpy2D(rhs, NX*sizeof(Real), rhs_gpu, 2*MX_hat*sizeof(Real),
    NX*sizeof(Real), NY, cudaMemcpyDeviceToHost);
  cudaMemset(rhs_gpu, 0, MY * MX_hat * 2 * sizeof(Real) );
  //t4.stop();
  //printf("%f %f %f %f %f\n",t0.get(),t1.get(),t2.get(),t3.get(),t4.get());
}

__global__ void kGreen(const int NX, const int NY, const int MX, const int MY,
  const int MX_hat, const Real fac, const Real h, Real*const in_out)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= MX) || (j >= MY) ) return;
  const Real xi = i>=NX? MX-i : i, yi = j>=NY? MY-j : j;
  const Real r = std::sqrt(xi*xi + yi*yi);
  if(r > 0) in_out[i +2*MX_hat*j] = fac * std::log(h * r);
  // r_eq = h / sqrt(pi)
  // G = 1/4 * r_eq^2 * (2* ln(r_eq) - 1)
  else      in_out[i +2*MX_hat*j] = fac/2 * (2*std::log(h/std::sqrt(M_PI)) - 1);
}

__global__ void kCopyC2R(const int MY, const int MX_hat, const Real norm,
  const Cmpl*const G_hat, Real*const m_kernel)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ( (i >= MX_hat) || (j >= MY) ) return;
  m_kernel[i + MX_hat*j] = G_hat[i + MX_hat*j].x * norm;
}

void clearCuMem(Real * buf, const size_t size) { cudaMemset(buf, 0, size ); }

void initGreen(const int NY, const int NX, const Real h, Real*const m_kernel)
{
  const int MX = 2 * NX - 1, MY = 2 * NY - 1, MX_hat = MX/2 +1;
  Real * tmp;
  cudaMalloc((void **)& tmp, MY * MX_hat * sizeof(Cmpl) );
  {
    const Real fac = 1 / ( 2*M_PI );
    dim3 dimB(16, 16), dimG( (MX+1) / dimB.x, (MY+1) / dimB.y);
    assert(((MX+1) % dimB.x == 0) && ((MY+1) % dimB.y == 0));
    kGreen<<<dimG, dimB>>> (NX,NY, MX,MY, MX_hat, fac, h, tmp);
  }
  {
    cufftHandle fwd;
    cufftPlan2d(&fwd, MY, MX, cufftPlanFWD);
    cufftExecFWD(fwd, tmp, (Cmpl*) tmp);
    cufftDestroy(fwd);
  }
  {
    const Real norm = 1.0 / (MX * MY);
    dim3 dimB(16, 16), dimG( MX_hat / dimB.x, (MY+1) / dimB.y);
    assert((MX_hat % dimB.x == 0) && ((MY+1) % dimB.y == 0));
    kCopyC2R<<<dimG, dimB>>> (MY, MX_hat, norm, (Cmpl*)tmp, m_kernel);
  }
  cudaFree(tmp);
}
