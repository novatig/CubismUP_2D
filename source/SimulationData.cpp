#include "Definitions.h"
#include "Shape.h"
#include "Operators/IC.h"
#include "Operators/Helpers.h"

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
}

void SimulationData::resetAll()
{
  for(const auto& shape : shapes) shape->resetAll();
  this->time = 0;
  step = 0;
  uinfx = 0;
  uinfy = 0;
  nextDumpTime = 0;
  _bDump = false;
  bPing = false;
  IC ic(*this);
  ic(0);
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
    maxS = std::max(maxS, (double) std::fabs( shape->getU() ) );
    maxS = std::max(maxS, (double) std::fabs( shape->getV() ) );
  }
  return maxS;
}

double SimulationData::maxRelSpeed() const
{
  double maxS = 0;
  for(const auto& shape : shapes) {
    maxS = std::max(maxS, (double) std::fabs(shape->getU() + uinfx ));
    maxS = std::max(maxS, (double) std::fabs(shape->getV() + uinfy ));
  }
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
  #ifndef SMARTIES_APP
    profiler->push_start(name);
  #endif
}
void SimulationData::stopProfiler()
{
  #ifndef SMARTIES_APP
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
