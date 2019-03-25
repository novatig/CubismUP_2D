//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Definitions.h"
#include "Shape.h"
#include "Operators/Helpers.h"

#include <HDF5Dumper.h>
void SimulationData::allocateGrid()
{
  chi   = new ScalarGrid(bpdx, bpdy, 1);
  vel   = new VectorGrid(bpdx, bpdy, 1);
  uDef  = new VectorGrid(bpdx, bpdy, 1);
  pres  = new ScalarGrid(bpdx, bpdy, 1);
  force = new VectorGrid(bpdx, bpdy, 1);

  pRHS  = new ScalarGrid(bpdx, bpdy, 1);
  tmpV  = new VectorGrid(bpdx, bpdy, 1);
  tmp   = new ScalarGrid(bpdx, bpdy, 1);
  DdF   = new VectorGrid(bpdx, bpdy, 1);

  invRho= new ScalarGrid(bpdx, bpdy, 1);
  pOld  = new ScalarGrid(bpdx, bpdy, 1);

  dump  = new DumpGrid(bpdx, bpdy, 1);
}

void SimulationData::dumpGlue(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<DumpGrid,StreamerGlue>(*(dump), step, time,
    "velChi_" + ss.str(), path4serialization);
}
void SimulationData::dumpChi(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<ScalarGrid,StreamerScalar>(*(chi), step, time,
    "chi_" + ss.str(), path4serialization);
}
void SimulationData::dumpPres(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<ScalarGrid,StreamerScalar>(*(pres), step, time,
    "pres_" + ss.str(), path4serialization);
}
void SimulationData::dumpPrhs(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<ScalarGrid,StreamerScalar>(*(pRHS), step, time,
    "pRHS_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmp(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<ScalarGrid,StreamerScalar>(*(tmp), step, time,
    "tmp_" + ss.str(), path4serialization);
}
void SimulationData::dumpVel(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<VectorGrid,StreamerVector>(*(vel), step, time,
    "vel_" + ss.str(), path4serialization);
}
void SimulationData::dumpUobj(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<VectorGrid,StreamerVector>(*(uDef), step, time,
    "uobj_" + ss.str(), path4serialization);
}
void SimulationData::dumpForce(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<VectorGrid,StreamerVector>(*(force), step, time,
    "force_" + ss.str(), path4serialization);
}
void SimulationData::dumpTmpV(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<VectorGrid,StreamerVector>(*(tmpV), step, time,
    "tmpV_" + ss.str(), path4serialization);
}
void SimulationData::dumpInvRho(std::string name) {
  std::stringstream ss; ss<<name<<std::setfill('0')<<std::setw(7)<<step;
  DumpHDF5<ScalarGrid,StreamerScalar>(*(invRho), step, time,
    "invRho_" + ss.str(), path4serialization);
}

void SimulationData::resetAll()
{
  for(const auto& shape : shapes) shape->resetAll();
  time = 0;
  step = 0;
  uinfx = 0;
  uinfy = 0;
  nextDumpTime = 0;
  _bDump = false;
  bPing = false;
}

void SimulationData::registerDump()
{
  nextDumpTime += dumpTime;
}

double SimulationData::minRho() const
{
  double minR = 1; // fluid is 1
  for(const auto& shape : shapes)
    minR = std::min( (double) shape->getMinRhoS(), minR );
  return minR;
}

void SimulationData::checkVariableDensity()
{
  bVariableDensity = false;
  for(const auto& shape : shapes)
    bVariableDensity = bVariableDensity || shape->bVariableDensity();
  if( bVariableDensity) std::cout << "Using variable density solver\n";
  if(!bVariableDensity) std::cout << "Using constant density solver\n";
}

double SimulationData::maxSpeed() const
{
  double maxS = 0;
  for(const auto& shape : shapes) {
    maxS = std::max(maxS, (double) shape->getMaxVel() );
  }
  return maxS;
}

double SimulationData::maxRelSpeed() const
{
  double maxS = 0;
  for(const auto& shape : shapes)
    maxS = std::max(maxS, (double) shape->getMaxVel() );
  return maxS;
}

SimulationData::~SimulationData()
{
  #ifndef SMARTIES_APP
    delete profiler;
  #endif
  if(vel not_eq nullptr) delete vel;
  if(chi not_eq nullptr) delete chi;
  if(uDef not_eq nullptr) delete uDef;
  if(pres not_eq nullptr) delete pres;
  if(force not_eq nullptr) delete force;
  if(pRHS not_eq nullptr) delete pRHS;
  if(tmpV not_eq nullptr) delete tmpV;
  if(invRho not_eq nullptr) delete invRho;
  if(pOld not_eq nullptr) delete pOld;
  if(tmp not_eq nullptr) delete tmp;
  while( not shapes.empty() ) {
    Shape * s = shapes.back();
    if(s not_eq nullptr) delete s;
    shapes.pop_back();
  }
}

bool SimulationData::bOver() const
{
  const bool timeEnd = endTime>0 && time >= endTime;
  const bool stepEnd =  nsteps>0 && step > nsteps;
  return timeEnd || stepEnd;
}

bool SimulationData::bDump()
{
  const bool timeDump = dumpTime>0 && time >= nextDumpTime;
  const bool stepDump = dumpFreq>0 && (step % dumpFreq) == 0;
  _bDump = stepDump || timeDump;
  return _bDump;
}

void SimulationData::startProfiler(std::string name)
{
  //std::cout << name << std::endl;
  Checker check (*this);
  check.run("before" + name);

  #ifndef SMARTIES_APP
    profiler->push_start(name);
  #endif
}
void SimulationData::stopProfiler()
{
  #ifndef SMARTIES_APP
    Checker check (*this);
    check.run("after" + profiler->currentAgentName());
    profiler->pop_stop();
  #endif
}
void SimulationData::printResetProfiler()
{
  #ifndef SMARTIES_APP
    profiler->printSummary();
    profiler->reset();
  #endif
}

void SimulationData::dumpAll(std::string name)
{
  const std::vector<BlockInfo>& chiInfo = chi->getBlocksInfo();
  const std::vector<BlockInfo>& velInfo = vel->getBlocksInfo();
  const std::vector<BlockInfo>& dmpInfo =dump->getBlocksInfo();
  #pragma omp parallel for schedule(static)
  for (size_t i=0; i < velInfo.size(); i++)
  {
    VectorBlock* VEL = (VectorBlock*) velInfo[i].ptrBlock;
    ScalarBlock* CHI = (ScalarBlock*) chiInfo[i].ptrBlock;
    VelChiGlueBlock& DMP = * (VelChiGlueBlock*) dmpInfo[i].ptrBlock;
    DMP.assign(CHI, VEL); // TODO USER MIGHT WANT TO HAVE invRho instead of chi?
  }
  //dumpChi  (name); // glued together: skip
  //dumpVel  (name); // glued together: skip
  dumpGlue(name);
  dumpPres (name);
  dumpInvRho (name);
  //dumpTmp  (name); // usually signed dist is here
  //dumpUobj (name);
  //dumpForce(name);
  //dumpTmpV (name); // probably useless
}
