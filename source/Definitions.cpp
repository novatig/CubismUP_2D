#include "Definitions.h"
#include "Shape.h"
void SimulationData::registerDump()
{
  nextDumpTime += dumpTime;
  shapes[0]->characteristic_function();
}
double SimulationData::minRho() const
{
  double minR = 1; // fluid is 1
  minR = std::min( (double) shapes[0]->getMinRhoS(), minR );
  return minR;
}
double SimulationData::maxSpeed() const
{
  double maxS = 0;
  maxS = std::max(maxS, (double) std::fabs( shapes[0]->getU() ) );
  maxS = std::max(maxS, (double) std::fabs( shapes[0]->getV() ) );
  return maxS;
}
double SimulationData::maxRelSpeed() const
{
  double maxS = 0;
  maxS = std::max(maxS, (double) std::fabs(shapes[0]->getU() + uinfx ));
  maxS = std::max(maxS, (double) std::fabs(shapes[0]->getV() + uinfy ));
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
