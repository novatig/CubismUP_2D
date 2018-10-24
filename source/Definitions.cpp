#include "Definitions.h"
#include "Shape.h"
#include "CoordinatorIC.h"
#include "HelperOperators.h"

void SimulationData::resetAll() {
  for(const auto& shape : shapes) shape->resetAll();
  this->time = 0;
  step = 0;
  uinfx = 0;
  uinfy = 0;
  nextDumpTime = 0;
  _bDump = false;
  bPing = false;
  CoordinatorIC coordIC(*this);
  coordIC(0);
}
void SimulationData::registerDump()
{
  nextDumpTime += dumpTime;
  for(const auto& shape : shapes) shape->characteristic_function();
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
  if(grid not_eq nullptr) delete grid;
  while( not shapes.empty() ) {
    Shape * s = shapes.back();
    if(s not_eq nullptr) delete s;
    shapes.pop_back();
  }
}
