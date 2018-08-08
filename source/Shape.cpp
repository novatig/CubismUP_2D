//
//  Shape.h
//  CubismUP_2D
//
//  Virtual shape class which defines the interface
//  Default simple geometries are also provided and can be used as references
//
//  This class only contains static information (position, orientation,...), no dynamics are included (e.g. velocities,...)
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Shape.h"
#include "OperatorComputeForces.h"

Real Shape::getMinRhoS() const { return rhoS; }
bool Shape::bVariableDensity() const {
  return std::fabs(rhoS-1.0) > numeric_limits<Real>::epsilon();
}

void Shape::updatePosition(double dt)
{
  // update centerOfMass - this is the reference point from which we compute the center
  // Remember, uinf is -ubox, therefore we sum it to u body to get
  // velocity of shapre relative to the sim box
  centerOfMass[0] += dt*( u + sim.uinfx );
  centerOfMass[1] += dt*( v + sim.uinfy );
  labCenterOfMass[0] += dt*u;
  labCenterOfMass[1] += dt*v;

  orientation += dt*omega;
  orientation = orientation>2*M_PI ? orientation-2*M_PI : (orientation<0 ? orientation+2*M_PI : orientation);

  center[0] = centerOfMass[0] + std::cos(orientation)*d_gm[0] - std::sin(orientation)*d_gm[1];
  center[1] = centerOfMass[1] + std::sin(orientation)*d_gm[0] + std::cos(orientation)*d_gm[1];
}

void Shape::outputSettings(ostream &outStream) const
{
  outStream << "centerX " << center[0] << endl;
  outStream << "centerY " << center[1] << endl;
  outStream << "centerMassX " << centerOfMass[0] << endl;
  outStream << "centerMassY " << centerOfMass[1] << endl;
  outStream << "orientation " << orientation << endl;
  outStream << "rhoS " << rhoS << endl;
}

Shape::Integrals Shape::integrateObstBlock(const vector<BlockInfo>& vInfo)
{
  double _m=0, _V=0, _j=0, _u=0, _v=0, _a=0, _x=0, _y=0, _X=0, _Y=0;
  //numerical trick to improve accuracy of sum: multiply m by h!
  const Real h = vInfo[0].h_gridpoint;
  #pragma omp parallel for schedule(dynamic) reduction( +: _m,_V,_x,_y,_X,_Y )
  for(size_t i=0; i<vInfo.size(); i++) {
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real chi = pos->second->chi[iy][ix]*h;
      if (chi <= 0) continue;
      Real p[2];
      vInfo[i].pos(p, ix, iy);
      const Real rhochi = chi*pos->second->rho[iy][ix];
      _x += rhochi*p[0];
      _y += rhochi*p[1];
      _X += chi*p[0];
      _Y += chi*p[1];
      _V += chi;
      _m += rhochi;
    }
  }
  //first compute the center of mass:
  _x /= _m; _y /= _m; _X /= _V; _Y /= _V;
  // here for numerical acuracy don't premultiply by h
  #pragma omp parallel for schedule(dynamic) reduction( + :_j, _u, _v, _a )
  for(size_t i=0; i<vInfo.size(); i++) {
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real chi = pos->second->chi[iy][ix];
      if (chi <= 0) continue;
      Real p[2];
      vInfo[i].pos(p, ix,iy);
      p[0] -= _x; p[1] -= _y;
      const Real u_ = pos->second->udef[iy][ix][0];
      const Real v_ = pos->second->udef[iy][ix][1];
      const Real rhochi = chi*pos->second->rho[iy][ix];
      _u += rhochi*u_;
      _v += rhochi*v_;
      _a += rhochi*(p[0]*v_ - p[1]*u_);
      _j += rhochi*(p[0]*p[0] + p[1]*p[1]);
    }
  }
  //divide by mass/inertia to get velocities
  _u *= h/_m; _v *= h/_m; _a /= _j;
  // multiply by grid-area to get proper mass , moment , and volume
  _m *= h; _j *= std::pow(h, 2); _V *= h;
  Shape::Integrals I(_m, _V, _j, _u, _v, _a, _x, _y, _X, _Y);
  return I;
}

void Shape::removeMoments(const vector<BlockInfo>& vInfo)
{
  Shape::Integrals I = integrateObstBlock(vInfo);
  #ifndef RL_TRAIN
  if(sim.verbose)
  if(fabs(I.u)+fabs(I.v)+fabs(I.a)>10*numeric_limits<Real>::epsilon())
    printf("Correction of: lin mom [%f %f] ang mom [%f]. Error in CM=[%f %f]\n", I.u, I.v, I.a, I.x-centerOfMass[0], I.y-centerOfMass[1]);
  #endif
  //update the center of mass, this operation should not move 'center'
  //cout << centerOfMass[0]<<" "<<I.x<<" "<<centerOfMass[1]<<" "<<I.y << endl;
  //cout << center[0]<<" "<<I.X<<" "<<center[1]<<" "<<I.Y << endl;
  centerOfMass[0] = I.x; centerOfMass[1] = I.y;
  //center[0] = I.X; center[1] = I.Y;
  const Real dCx = center[0]-centerOfMass[0];
  const Real dCy = center[1]-centerOfMass[1];
  d_gm[0] =  dCx*std::cos(orientation) +dCy*std::sin(orientation);
  d_gm[1] = -dCx*std::sin(orientation) +dCy*std::cos(orientation);

  #ifndef NDEBUG
    Real Cxtest = center[0] -std::cos(orientation)*d_gm[0] + std::sin(orientation)*d_gm[1];
    Real Cytest = center[1] -std::sin(orientation)*d_gm[0] - std::cos(orientation)*d_gm[1];
    if( std::abs(Cxtest-centerOfMass[0])>numeric_limits<Real>::epsilon() ||
        std::abs(Cytest-centerOfMass[1])>numeric_limits<Real>::epsilon() ) {
        printf("Error update of center of mass = [%f %f]\n", Cxtest-centerOfMass[0], Cytest-centerOfMass[1]);
          abort();
        }
  #endif

  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];
        pos->second->udef[iy][ix][0] -= I.u -I.a*p[1];
        pos->second->udef[iy][ix][1] -= I.v +I.a*p[0];
    }
  }

  #ifndef NDEBUG
   Shape::Integrals Itest = integrateObstBlock(vInfo);
   if(std::abs(Itest.u)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.v)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.a)>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.x-centerOfMass[0])>10*numeric_limits<Real>::epsilon() ||
      std::abs(Itest.y-centerOfMass[1])>10*numeric_limits<Real>::epsilon() ){
    printf("After correction: linm [%e %e] angm [%e] deltaCM=[%e %e]\n",
    Itest.u,Itest.v,Itest.a,Itest.x-centerOfMass[0],Itest.y-centerOfMass[1]);
    abort();
   }
  #endif
};

void Shape::computeVelocities()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  Real _M = 0, _V = 0, _J = 0, um = 0, vm = 0, am = 0; //linear momenta
  #pragma omp parallel for schedule(dynamic) reduction(+:_M,_V,_J,um,vm,am)
  for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real chi = pos->second->chi[iy][ix];
        if (chi <= 0) continue;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0] -= centerOfMass[0];
        p[1] -= centerOfMass[1];

        const Real rhochi = pos->second->rho[iy][ix] * chi;
        _M += rhochi; _V += chi;
        um += rhochi * b(ix,iy).u;
        vm += rhochi * b(ix,iy).v;
        _J += rhochi * (p[0]*p[0]       + p[1]*p[1]);
        am += rhochi * (p[0]*b(ix,iy).v - p[1]*b(ix,iy).u);
      }
  }

  computedu = um / _M;
  computedv = vm / _M;
  computedo = am / _J;
  if(bForcedx) u = forcedu;
  else         u = computedu;

  if(bForcedy) v = forcedv;
  else         v = computedv;

  if(not bBlockang) omega = computedo;

  J = _J * std::pow(vInfo[0].h_gridpoint,2);
  M = _M * std::pow(vInfo[0].h_gridpoint,2);
  V = _V * std::pow(vInfo[0].h_gridpoint,2);

  #ifndef RL_TRAIN
    if(sim.verbose)
      printf("CM:[%f %f] C:[%f %f] u:%f v:%f omega:%f M:%f J:%f V:%f\n",
      centerOfMass[0], centerOfMass[1], center[0], center[1], u, v, omega, M, J, V);
    if(not sim.muteAll)
    {
      ofstream fileSpeed;
      stringstream ssF;
      ssF<<"velocity_"<<obstacleID<<".dat";
      fileSpeed.open(ssF.str().c_str(), ios::app);
      if(sim.step==0)
        fileSpeed<<"time dt CMx CMy angle u v omega M J accx accy"<<std::endl;

      fileSpeed<<sim.time<<" "<<sim.dt<<" "<<centerOfMass[0]<<" "<<centerOfMass[1]<<" "<<orientation<<" "<<u <<" "<<v<<" "<<omega <<" "<<M<<" "<<J<<endl;
      fileSpeed.close();
    }
  #endif
}

void Shape::updateLabVelocity( int nSum[2], double uSum[2] )
{
  if(bFixedx) { (nSum[0])++; uSum[0] -= u; }
  if(bFixedy) { (nSum[1])++; uSum[1] -= v; }
}

void Shape::penalize()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  const Real lamdt = sim.dt * sim.lambda;
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock & b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real chi = pos->second->chi[iy][ix];
      if (chi <= 0) continue;
      Real p[2];
      vInfo[i].pos(p, ix, iy);
      p[0] -= centerOfMass[0]; p[1] -= centerOfMass[1];
      const Real alpha = 1/(1 + lamdt * chi);
      const Real*const udef = pos->second->udef[iy][ix];
      const Real uTot = u -omega*p[1] +udef[0];// -sim.uinfx;
      const Real vTot = v +omega*p[0] +udef[1];// -sim.uinfy;
      b(ix,iy).u = alpha*b(ix,iy).u + (1-alpha)*uTot;
      b(ix,iy).v = alpha*b(ix,iy).v + (1-alpha)*vTot;
    }
  }
}

void Shape::diagnostics()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  Real _a=0,_m=0,_x=0,_y=0,_t=0;
  #pragma omp parallel for schedule(dynamic) reduction(+:_a,_m,_x,_y,_t)
  for(size_t i=0; i<vInfo.size(); i++) {
      const auto pos = obstacleBlocks.find(vInfo[i].blockID);
      if(pos == obstacleBlocks.end()) continue;
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;

      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real Xs = pos->second->chi[iy][ix];
        if (Xs <= 0) continue;
        Real p[2];
        vInfo[i].pos(p, ix, iy);
        p[0]-=centerOfMass[0];
        p[1]-=centerOfMass[1];
        const Real*const udef = pos->second->udef[iy][ix];
        const Real uDiff = b(ix,iy).u - (u -omega*p[1] +udef[0]);
        const Real vDiff = b(ix,iy).v - (v +omega*p[0] +udef[1]);
        _a += Xs;
        _m += Xs / b(ix,iy).invRho;
        _x += uDiff*Xs;
        _y += vDiff*Xs;
        _t += (p[0]*vDiff-p[1]*uDiff)*Xs;
      }
  }
  const double dV = std::pow(vInfo[0].h_gridpoint, 2);
  area_penal   = _a * dV;
  mass_penal   = _m * dV;
  forcex_penal = _x * dV * sim.lambda;
  forcey_penal = _y * dV * sim.lambda;
  torque_penal = _t * dV * sim.lambda;
}

void Shape::characteristic_function()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      b(ix,iy).tmp = pos->second->chi[iy][ix];
  }
}

void Shape::deformation_velocities()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<vInfo.size(); i++) {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    const auto pos = obstacleBlocks.find(vInfo[i].blockID);
    if(pos == obstacleBlocks.end()) continue;

    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      b(ix,iy).tmpU += pos->second->udef[iy][ix][0];
      b(ix,iy).tmpV += pos->second->udef[iy][ix][1];
    }
  }
}

void Shape::computeForces()
{
  const vector<BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  Real vel_unit[2] = {0., 0.};
  const Real vel_norm = std::sqrt(u*u + v*v);
  if(vel_norm>0){ vel_unit[0]=u/vel_norm; vel_unit[1]=v/vel_norm; }

  OperatorComputeForces finalize(sim.nu, vel_unit, centerOfMass);
  compute_surface(finalize, vInfo);

  //additive quantities:
  perimeter = 0; forcex = 0; forcey = 0; forcex_P = 0;
  forcey_P = 0; forcex_V = 0; forcey_V = 0; torque = 0;
  torque_P = 0; torque_V = 0; drag = 0; thrust = 0;
  Pout = 0; PoutBnd = 0; defPower = 0; defPowerBnd = 0; circulation = 0;

  for (auto & block : obstacleBlocks)
  {
    circulation += block.second->circulation;
    perimeter += block.second->perimeter; torque   += block.second->torque;
    forcex   += block.second->forcex;     forcey   += block.second->forcey;
    forcex_P += block.second->forcex_P;   forcey_P += block.second->forcey_P;
    forcex_V += block.second->forcex_V;   forcey_V += block.second->forcey_V;
    torque_P += block.second->torque_P;   torque_V += block.second->torque_V;
    drag     += block.second->drag;       thrust   += block.second->thrust;
    Pout += block.second->Pout; defPowerBnd += block.second->defPowerBnd;
    PoutBnd += block.second->PoutBnd; defPower += block.second->defPower;
  }

  //derived quantities:
  Pthrust    = thrust*vel_norm;
  Pdrag      =   drag*vel_norm;
  EffPDef    = Pthrust/(Pthrust-min(defPower,(Real)0.));
  EffPDefBnd = Pthrust/(Pthrust-    defPowerBnd);

  #ifndef RL_TRAIN
  if (sim._bDump && not sim.muteAll) {
    char buf[500];
    sprintf(buf, "surface_%01d_%07d.raw", obstacleID, sim.step);
    FILE * pFile = fopen (buf, "wb");
    for(auto & block : obstacleBlocks) block.second->print(pFile);
    fflush(pFile);
    fclose(pFile);
  }
  if(not sim.muteAll)
  {
    ofstream fileForce;
    ofstream filePower;
    stringstream ssF, ssP;
    ssF<<"forceValues_"<<obstacleID<<".dat";
    ssP<<"powerValues_"<<obstacleID<<".dat"; //obstacleID

    fileForce.open(ssF.str().c_str(), ios::app);
    if(sim.step==0)
      fileForce<<"time Fx Fy FxPres FyPres FxVisc FyVisc tau tauPres tauVisc drag thrust perimeter circulation area_penal mass_penal forcex_penal forcey_penal torque_penal"<<std::endl;

    fileForce<<sim.time<<" "<<forcex<<" "<<forcey<<" "<<forcex_P<<" "<<forcey_P<<" "<<forcex_V <<" "<<forcey_V<<" "<<torque <<" "<<torque_P<<" "<<torque_V<<" "<<drag<<" "<<thrust<<" "<<perimeter<<" "<<circulation<<" "<<area_penal<<" "<<mass_penal<<" "<<forcex_penal<<" "<<forcey_penal<<" "<<torque_penal<<endl;
    fileForce.close();

    filePower.open(ssP.str().c_str(), ios::app);
    if(sim.step==0)
      filePower<<"time Pthrust Pdrag PoutBnd Pout defPowerBnd defPower EffPDefBnd EffPDef"<<std::endl;
    filePower<<sim.time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<endl;
    filePower.close();
  }
  #endif
}
