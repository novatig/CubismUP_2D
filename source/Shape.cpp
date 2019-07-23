//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Shape.h"
//#include "OperatorComputeForces.h"
#include "Utils/BufferedLogger.h"
#include <gsl/gsl_linalg.h>
#include <iomanip>
using namespace cubism;

static constexpr double EPS = std::numeric_limits<double>::epsilon();
Real Shape::getMinRhoS() const { return rhoS; }
Real Shape::getCharMass() const { return 0; }
Real Shape::getMaxVel() const { return std::sqrt(u*u + v*v); }
bool Shape::bVariableDensity() const {
  return std::fabs(rhoS-1.0) > std::numeric_limits<Real>::epsilon();
}

void Shape::updateVelocity(double dt)
{
  double A[3][3] = {
    {   penalM,       0, -penalDY },
    {        0,  penalM,  penalDX },
    { -penalDY, penalDX,  penalJ  }
  };
  double b[3] = {
    fluidMomX   + dt * appliedForceX,
    fluidMomY   + dt * appliedForceY,
    fluidAngMom + dt * appliedTorque
  };

  if(bForcedx) {
                 A[0][1] = 0; A[0][2] = 0; b[0] = penalM * forcedu;
  }
  if(bForcedy) {
    A[1][0] = 0;              A[1][2] = 0; b[1] = penalM * forcedv;
  }
  if(bBlockang) {
    A[2][0] = 0; A[2][1] = 0;              b[2] = penalJ * forcedomega;
  }

  gsl_matrix_view Agsl = gsl_matrix_view_array (&A[0][0], 3, 3);
  gsl_vector_view bgsl = gsl_vector_view_array (b, 3);
  gsl_vector * xgsl = gsl_vector_alloc (3);
  int sgsl;
  gsl_permutation * permgsl = gsl_permutation_alloc (3);
  gsl_linalg_LU_decomp (& Agsl.matrix, permgsl, & sgsl);
  gsl_linalg_LU_solve (& Agsl.matrix, permgsl, & bgsl.vector, xgsl);

  if(not bForcedy)  u     = gsl_vector_get(xgsl, 0);
  if(not bForcedy)  v     = gsl_vector_get(xgsl, 1);
  if(not bBlockang) omega = gsl_vector_get(xgsl, 2);
}

/*
if(not bForcedx) {
  const Real uNxt = fluidMomX / penalM;
  const Real FX = M * (uNxt - u) / dt;
  const Real accx = ( appliedForceX + FX ) / M;
  u = u + dt * accx;
}

if(not bForcedy) {
  const Real vNxt = fluidMomY / penalM;
  const Real FY = M * (vNxt - v) / dt;
  const Real accy = ( appliedForceY + FY ) / M;
  v = v + dt * accy;
}

if(not bBlockang) {
  const Real omegaNxt = fluidAngMom / penalJ;
  const Real TZ = J * (omegaNxt - omega) / dt;
  const Real acca = ( appliedTorque + TZ ) / J;
  omega = omega + dt * acca;
}
*/

void Shape::updateLabVelocity( int nSum[2], double uSum[2] )
{
  if(bFixedx) { (nSum[0])++; uSum[0] -= u; }
  if(bFixedy) { (nSum[1])++; uSum[1] -= v; }
}

void Shape::updatePosition(double dt)
{
  // Remember, uinf is -ubox, therefore we sum it to u body to get
  // velocity of shapre relative to the sim box
  centerOfMass[0] += dt * ( u + sim.uinfx );
  centerOfMass[1] += dt * ( v + sim.uinfy );
  labCenterOfMass[0] += dt * u;
  labCenterOfMass[1] += dt * v;

  orientation += dt*omega;
  orientation = orientation> M_PI ? orientation-2*M_PI : orientation;
  orientation = orientation<-M_PI ? orientation+2*M_PI : orientation;

  const double cosang = std::cos(orientation), sinang = std::sin(orientation);

  center[0] = centerOfMass[0] + cosang*d_gm[0] - sinang*d_gm[1];
  center[1] = centerOfMass[1] + sinang*d_gm[0] + cosang*d_gm[1];

  const Real CX = labCenterOfMass[0], CY = labCenterOfMass[1], t = sim.time;
  const Real cx = centerOfMass[0], cy = centerOfMass[1], angle = orientation;

  if(sim.dt <= 0) return;

  if(sim.verbose)
    printf("CM:[%.02f %.02f] C:[%.02f %.02f] ang:%.02f u:%.03f v:%.03f av:%.03f"
      " M:%.02e J:%.02e\n", cx, cy, center[0], center[1], angle, u, v, omega, M, J);
  if(not sim.muteAll)
  {
    std::stringstream ssF;
    ssF<<sim.path2file<<"/velocity_"<<obstacleID<<".dat";
    std::stringstream & fout = logger.get_stream(ssF.str());
    if(sim.step==0)
     fout<<"t dt CXsim CYsim CXlab CYlab angle u v omega M J accx accy accw\n";

    fout<<t<<" "<<dt<<" "<<cx<<" "<<cy<<" "<<CX<<" "<<CY<<" "<<angle<<" "
        <<u<<" "<<v<<" "<<omega<<" "<<M<<" "<<J<<" "<<fluidMomX/penalM<<" "
        <<fluidMomY/penalM<<" "<<fluidAngMom/penalJ<<"\n";
  }
}

void Shape::outputSettings(std::ostream &outStream) const
{
  outStream << "centerX " << center[0] << "\n";
  outStream << "centerY " << center[1] << "\n";
  outStream << "centerMassX " << centerOfMass[0] << "\n";
  outStream << "centerMassY " << centerOfMass[1] << "\n";
  outStream << "orientation " << orientation << "\n";
  outStream << "rhoS " << rhoS << "\n";
}

Shape::Integrals Shape::integrateObstBlock(const std::vector<BlockInfo>& vInfo)
{
  double _x=0, _y=0, _m=0, _j=0, _u=0, _v=0, _a=0;
  const double hsq = std::pow(vInfo[0].h_gridpoint, 2);
  #pragma omp parallel for schedule(dynamic,1) reduction(+:_x,_y,_m,_j,_u,_v,_a)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if(pos == nullptr) continue;
    const CHI_MAT & __restrict__ CHI = pos->chi;
    const CHI_MAT & __restrict__ RHO = pos->rho;
    const UDEFMAT & __restrict__ UDEF = pos->udef;
    for(int iy=0; iy<ObstacleBlock::sizeY; ++iy)
    for(int ix=0; ix<ObstacleBlock::sizeX; ++ix)
    {
      if (CHI[iy][ix] <= 0) continue;
      double p[2];
      vInfo[i].pos(p, ix, iy);
      const double rhochi = CHI[iy][ix] * RHO[iy][ix] * hsq;
      p[0] -= centerOfMass[0];
      p[1] -= centerOfMass[1];
      _x += rhochi*p[0];
      _y += rhochi*p[1];
      _m += rhochi;
      _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
      _u += rhochi*UDEF[iy][ix][0];
      _v += rhochi*UDEF[iy][ix][1];
      _a += rhochi*(p[0]*UDEF[iy][ix][1] - p[1]*UDEF[iy][ix][0]);
    }
  }
  assert(std::fabs(_x)     < 10*std::numeric_limits<Real>::epsilon() );
  assert(std::fabs(_y)     < 10*std::numeric_limits<Real>::epsilon() );
  assert(std::fabs(M - _m) < 10*std::numeric_limits<Real>::epsilon() );
  _j = _j;
  _a = _a;
  // turn moments into velocities:
  _u /= _m;
  _v /= _m;
  _a /= _j;
  return Integrals(_x, _y, _m, _j, _u, _v, _a);
}

void Shape::removeMoments(const std::vector<BlockInfo>& vInfo)
{
  //static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  Shape::Integrals I = integrateObstBlock(vInfo);
  //if(sim.verbose)
    //if( std::max({std::fabs(I.u), std::fabs(I.v), std::fabs(I.a)}) > 10*EPS )
  //printf("Udef momenta: lin=[%e %e] ang=[%e]. Errors: dCM=[%e %e] dM=%e\n",
  //    I.u, I.v, I.a, I.x, I.y, std::fabs(I.m-M));
  M = I.m; J = I.j;

  //with current center put shape on grid, with current shape on grid we updated
  //the center of mass, now recompute the distance betweeen the two:
  const double dCx = center[0]-centerOfMass[0];
  const double dCy = center[1]-centerOfMass[1];
  d_gm[0] =  dCx*std::cos(orientation) +dCy*std::sin(orientation);
  d_gm[1] = -dCx*std::sin(orientation) +dCy*std::cos(orientation);

  #if 0 //ndef NDEBUG
    Real Cxtest = center[0] -std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
    Real Cytest = center[1] -std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
    if(std::fabs(Cxtest-centerOfMass[0])>EPS ||
       std::fabs(Cytest-centerOfMass[1])>EPS ) {
      printf("Error update of center of mass = [%f %f]\n",
        Cxtest-centerOfMass[0], Cytest-centerOfMass[1]); fflush(0); abort();
    }
  #endif

  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto pos = obstacleBlocks[vInfo[i].blockID];
    if(pos == nullptr) continue;

    for(int iy=0; iy<ObstacleBlock::sizeY; ++iy)
    for(int ix=0; ix<ObstacleBlock::sizeX; ++ix) {
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        pos->udef[iy][ix][0] -= I.u -I.a*p[1];
        pos->udef[iy][ix][1] -= I.v +I.a*p[0];
    }
  }

  #if 0 //ndef NDEBUG
   Shape::Integrals Itest = integrateObstBlock(vInfo);
   if( std::fabs(Itest.u)>1000*EPS || std::fabs(Itest.v)>1000*EPS ||
       std::fabs(Itest.x-centerOfMass[0])>1000*EPS ||
       std::fabs(Itest.y-centerOfMass[1])>1000*EPS ||
       std::fabs(Itest.a)>1000*EPS ) {
    printf("After correction: linm [%e %e] angm [%e] deltaCM=[%e %e]\n",
    Itest.u,Itest.v,Itest.a,Itest.x-centerOfMass[0],Itest.y-centerOfMass[1]);
    fflush(0); abort();
   }
  #endif
};

void Shape::diagnostics()
{
  /*
  const std::vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const double hsq = std::pow(vInfo[0].h_gridpoint, 2);
  double _a=0, _m=0, _x=0, _y=0, _t=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:_a,_m,_x,_y,_t)
  for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks[vInfo[i].blockID];
      if(pos == nullptr) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        if (pos->chi[iy][ix] <= 0) continue;
        const double Xs = pos->chi[iy][ix] * hsq;
        double p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        const Real*const udef = pos->udef[iy][ix];
        const double uDiff = b(ix,iy).u - (u -omega*p[1] +udef[0]);
        const double vDiff = b(ix,iy).v - (v +omega*p[0] +udef[1]);
        _a += Xs;
        _m += Xs / b(ix,iy).invRho;
        _x += uDiff*Xs;
        _y += vDiff*Xs;
        _t += (p[0]*vDiff-p[1]*uDiff)*Xs;
      }
  }
  area_penal   = _a;
  mass_penal   = _m;
  forcex_penal = _x * sim.lambda;
  forcey_penal = _y * sim.lambda;
  torque_penal = _t * sim.lambda;
  */
}

void Shape::computeForces()
{
  //additive quantities:
  perimeter = 0; forcex = 0; forcey = 0; forcex_P = 0;
  forcey_P = 0; forcex_V = 0; forcey_V = 0; torque = 0;
  torque_P = 0; torque_V = 0; drag = 0; thrust = 0;
  Pout = 0; PoutBnd = 0; defPower = 0; defPowerBnd = 0; circulation = 0;

  for (auto & block : obstacleBlocks) if(block not_eq nullptr)
  {
    circulation += block->circulation;
    perimeter += block->perimeter; torque   += block->torque;
    forcex   += block->forcex;     forcey   += block->forcey;
    forcex_P += block->forcex_P;   forcey_P += block->forcey_P;
    forcex_V += block->forcex_V;   forcey_V += block->forcey_V;
    torque_P += block->torque_P;   torque_V += block->torque_V;
    drag     += block->drag;       thrust   += block->thrust;
    Pout += block->Pout; defPowerBnd += block->defPowerBnd;
    PoutBnd += block->PoutBnd; defPower += block->defPower;
  }

  //derived quantities:
  Pthrust    = thrust * std::sqrt(u*u + v*v);
  Pdrag      =   drag * std::sqrt(u*u + v*v);
  const Real denUnb = Pthrust- std::min(defPower, (double)0);
  const Real demBnd = Pthrust-          defPowerBnd;
  EffPDef    = Pthrust/std::max(denUnb, EPS);
  EffPDefBnd = Pthrust/std::max(demBnd, EPS);

  if(sim.dt <= 0) return;

  if (sim._bDump && not sim.muteAll)
  {
    std::stringstream ssF; ssF<<sim.path2file<<"/surface_"<<obstacleID
      <<"_"<<std::setfill('0')<<std::setw(7)<<sim.step<<".raw";
    std::ofstream pFile(ssF.str().c_str(), std::ofstream::binary);
    for(auto & block : obstacleBlocks) if(block not_eq nullptr)
      block->print(pFile);
    pFile.close();
  }

  if(not sim.muteAll)
  {
    std::stringstream ssF, ssP;
    ssF<<sim.path2file<<"/forceValues_"<<obstacleID<<".dat";
    ssP<<sim.path2file<<"/powerValues_"<<obstacleID<<".dat"; //obstacleID

    std::stringstream &fileForce = logger.get_stream(ssF.str());
    if(sim.step==0)
      fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc"
                 " drag thrust perimeter circulation area_penal mass_penal"
                 " forcex_penal forcey_penal torque_penal\n";

    fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P
             <<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "
             <<torque_V<<" "<<drag<<" "<<thrust<<" "<<perimeter<<" "
             <<circulation<<"\n";

    std::stringstream &filePower = logger.get_stream(ssP.str());
    if(sim.step==0)
      filePower<<"time Pthrust Pdrag PoutBnd Pout defPowerBnd defPower"
                 " EffPDefBnd EffPDef\n";
    filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "
             <<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<"\n";
  }
}

Shape::Shape( SimulationData& s, ArgumentParser& p, double C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]}, orientation(origAng),
  rhoS( p("-rhoS").asDouble(1) ),
  bFixed(    p("-bFixed").asBool(false) ),
  bFixedx(   p("-bFixedx" ).asBool(bFixed) ),
  bFixedy(   p("-bFixedy" ).asBool(bFixed) ),
  bForced(   p("-bForced").asBool(false) ),
  bForcedx(  p("-bForcedx").asBool(bForced)),
  bForcedy(  p("-bForcedy").asBool(bForced)),
  bBlockang( p("-bBlockAng").asBool(bForcedx || bForcedy) ),
  forcedu( - p("-xvel").asDouble(0) ),
  forcedv( - p("-yvel").asDouble(0) ),
  forcedomega( - p("-angvel").asDouble(0) ) {  }

Shape::~Shape()
{
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
}
