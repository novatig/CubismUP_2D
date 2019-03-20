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

Real Shape::getMinRhoS() const { return rhoS; }
Real Shape::getCharMass() const { return 0; }
bool Shape::bVariableDensity() const {
  return std::fabs(rhoS-1.0) > std::numeric_limits<Real>::epsilon();
}

void Shape::updateVelocity(double dt)
{
  if(not bForcedx) u = fluidMomX / penalM;

  if(not bForcedy) v = fluidMomY / penalM;

  if(not bBlockang) omega = fluidAngMom / penalJ;
}

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

  if(sim.verbose)
    printf("CM:[%.02f %.02f] C:[%.02f %.02f] ang:%.02f u:%.03f v:%.03f av:%.03f"
      " M:%e J:%e\n", cx, cy, center[0], center[1], angle, u, v, omega, M, J);
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
      _x += rhochi*p[0];
      _y += rhochi*p[1];
      p[0] -= centerOfMass[0];
      p[1] -= centerOfMass[1];
      _m += rhochi;
      _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
      _u += rhochi*UDEF[iy][ix][0];
      _v += rhochi*UDEF[iy][ix][1];
      _a += rhochi*(p[0]*UDEF[iy][ix][1] - p[1]*UDEF[iy][ix][0]);
    }
  }
  _x /= _m;
  _y /= _m;
  // Parallel axis theorem:
  const double dC[2] = { _x - centerOfMass[0], _y - centerOfMass[1] };
  assert( std::fabs(dC[0]) < 1000*std::numeric_limits<Real>::epsilon() );
  assert( std::fabs(dC[1]) < 1000*std::numeric_limits<Real>::epsilon() );
  assert( std::fabs(M - _m) <  10*std::numeric_limits<Real>::epsilon() );

  // I_arbitrary_axis = I_CM + m * dist_CM_axis ^ 2 . Now _j is J around old CM
  _j = _j - _m*(dC[0]*dC[0] + dC[1]*dC[1]);
  _a = _a - ( dC[0]*_v - dC[1]*_u );
  // turn moments into velocities:
  _u /= _m;  _v /= _m;  _a /= _j;
  return Integrals(_x, _y, _m, _j, _u, _v, _a);
}

void Shape::removeMoments(const std::vector<BlockInfo>& vInfo)
{
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  Shape::Integrals I = integrateObstBlock(vInfo);
  M = I.m;
  J = I.j;
  if(sim.verbose)
    if( std::max({std::fabs(I.u), std::fabs(I.v), std::fabs(I.a)}) > 10*EPS )
      printf("Correct: lin mom [%f %f] ang mom [%f]. Error in CM=[%f %f]\n",
        I.u, I.v, I.a, I.x-centerOfMass[0], I.y-centerOfMass[1]);

  //update the center of mass, this operation should not move 'center'
  centerOfMass[0] = I.x; centerOfMass[1] = I.y;
  //center[0] = I.X; center[1] = I.Y;
  const double dCx = center[0]-centerOfMass[0];
  const double dCy = center[1]-centerOfMass[1];
  d_gm[0] =  dCx*std::cos(orientation) +dCy*std::sin(orientation);
  d_gm[1] = -dCx*std::sin(orientation) +dCy*std::cos(orientation);

  #ifndef NDEBUG
    Real Cxtest = center[0] -std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
    Real Cytest = center[1] -std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
    if(std::fabs(Cxtest-centerOfMass[0])>EPS ||
       std::fabs(Cytest-centerOfMass[1])>EPS ) {
      printf("Error update of center of mass = [%f %f]\n",
        Cxtest-centerOfMass[0], Cytest-centerOfMass[1]); fflush(0); abort();
    }
  #endif

  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
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

  #ifndef NDEBUG
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
  /*
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  Real vel_unit[2] = {0., 0.};
  const Real vel_norm = std::sqrt(u*u + v*v);
  if(vel_norm>0){ vel_unit[0]=u/vel_norm; vel_unit[1]=v/vel_norm; }

  OperatorComputeForces finalize(sim, vel_unit, centerOfMass);
  compute_surface(finalize, vInfo);

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
  Pthrust    = thrust*vel_norm;
  Pdrag      =   drag*vel_norm;
  EffPDef    = Pthrust/(Pthrust-min(defPower,(double)0));
  EffPDefBnd = Pthrust/(Pthrust-    defPowerBnd);

  if (sim._bDump && not sim.muteAll)
  {
    stringstream ssF; ssF<<sim.path2file<<"/surface_"<<obstacleID
      <<"_"<<std::setfill('0')<<std::setw(7)<<sim.step<<".raw";
    ofstream pFile(ssF.str().c_str(), ofstream::binary);
    for(auto & block : obstacleBlocks) if(block not_eq nullptr)
      block->print(pFile);
    pFile.close();
  }
  if(not sim.muteAll)
  {
    stringstream ssF, ssP;
    ssF<<sim.path2file<<"/forceValues_"<<obstacleID<<".dat";
    ssP<<sim.path2file<<"/powerValues_"<<obstacleID<<".dat"; //obstacleID

    std::stringstream &fileForce = logger.get_stream(ssF.str());
    if(sim.step==0)
      fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust perimeter circulation area_penal mass_penal forcex_penal forcey_penal torque_penal"<<std::endl;

    fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P<<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "<<torque_V<<" "<<drag<<" "<<thrust<<" "<<perimeter<<" "<<circulation<<" "<<area_penal<<" "<<mass_penal<<" "<<forcex_penal<<" "<<forcey_penal<<" "<<torque_penal<<endl;

    std::stringstream &filePower = logger.get_stream(ssP.str());
    if(sim.step==0)
      filePower<<"time Pthrust Pdrag PoutBnd Pout defPowerBnd defPower EffPDefBnd EffPDef"<<std::endl;
    filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<endl;
  }
  */
}

Shape::Shape( SimulationData& s, ArgumentParser& p, double C[2] ) :
  sim(s), origC{C[0],C[1]}, origAng( p("-angle").asDouble(0)*M_PI/180 ),
  center{C[0],C[1]}, centerOfMass{C[0],C[1]}, orientation(origAng),
  rhoS( p("-rhoS").asDouble(1) ),
  bForced( p("-bForced").asBool(false) ),
  bFixed( p("-bFixed").asBool(false) ),
  bForcedx(p("-bForcedx").asBool(bForced)),
  bForcedy(p("-bForcedy").asBool(bForced)),
  bBlockang( p("-bBlockAng").asBool(bForcedx || bForcedy) ),
  bFixedx(p("-bFixedx" ).asBool(bFixed) ),
  bFixedy(p("-bFixedy" ).asBool(bFixed) ),
  forcedu( - p("-xvel").asDouble(0) ),
  forcedv( - p("-yvel").asDouble(0) ) {  }

Shape::~Shape()
{
  for(auto & entry : obstacleBlocks) delete entry;
  obstacleBlocks.clear();
}
