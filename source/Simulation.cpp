//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#include "Simulation.h"

#include <Cubism/HDF5Dumper.h>
//#include <ZBinDumper.h>

#include "Operators/Helpers.h"
#include "Operators/PressureIterator.h"
#include "Operators/PressureSingle.h"
#include "Operators/PressureVarRho.h"
#include "Operators/PressureVarRho_proper.h"
#include "Operators/Penalization.h"
#include "Operators/PressureIterator_unif.h"
#include "Operators/PressureIterator_approx.h"
#include "Operators/PutObjectsOnGrid.h"
#include "Operators/UpdateObjects.h"
#include "Operators/advDiffGrav.h"
//#include "Operators/advDiff_RK.h"
#include "Operators/advDiff.h"

#include "Operators/presRHS_step1.h"
#include "Utils/FactoryFileLineParser.h"

#include "Obstacles/ShapesSimple.h"
#include "Obstacles/CarlingFish.h"
#include "Obstacles/StefanFish.h"
#include "Obstacles/BlowFish.h"
#include "Obstacles/SmartCylinder.h"
#include "Obstacles/Glider.h"

//#include <regex>
#include <algorithm>
#include <iterator>
using namespace cubism;

static inline std::vector<std::string> split(const std::string&s,const char dlm)
{
  std::stringstream ss(s); std::string item; std::vector<std::string> tokens;
  while (std::getline(ss, item, dlm)) tokens.push_back(item);
  return tokens;
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
 std::cout<<"===============================================================\n";
 std::cout<<"                  Flow past a falling obstacle                 \n";
 std::cout<<"===============================================================\n";
 parser.print_args();
}

Simulation::~Simulation()
{
  while( not pipeline.empty() ) {
    Operator * g = pipeline.back();
    pipeline.pop_back();
    if(g not_eq nullptr) delete g;
  }
}

void Simulation::parseRuntime()
{
  sim.bRestart = parser("-restart").asBool(false);
  std::cout << "bRestart is " << sim.bRestart << std::endl;

  // initialize grid
  parser.set_strict_mode();
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();
  parser.unset_strict_mode();
  sim.extent = parser("-extent").asDouble(1);
  sim.allocateGrid();

  // simulation ending parameters
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // output parameters
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);

  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);

  sim.poissonType = parser("-poissonType").asString("");
  // simulation settings
  sim.CFL = parser("-CFL").asDouble(.1);
  sim.lambda = parser("-lambda").asDouble(1e3 / sim.CFL);
  sim.dlm = parser("-dlm").asDouble(0);
  sim.nu = parser("-nu").asDouble(1e-2);

  sim.fadeLenX = parser("-fadeLen").asDouble(0);
  sim.fadeLenY = parser("-fadeLen").asDouble(0);

  sim.verbose = parser("-verbose").asInt(1);
  sim.iterativePenalization = parser("-iterativePenalization").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);//stronger silence, not even files
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes()
{
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors( shapeArg );
  std::string lines;
  unsigned k = 0;

  while (std::getline(descriptors, lines))
  {
    std::replace(lines.begin(), lines.end(), '_', ' ');
    // Two options! Either we have list of lines each containing a description
    // of an obstacle (like factory files or CUP3D factory-descriptor)
    // Or we have an argument list that looks like -shapes foo;bar. Splitter
    // will create a vector of strings, the first containing foo and the second
    // bar so that they can be parsed separately. Reason being that in many
    // situations \n will not be read as line escape but as backslash n.
    const std::vector<std::string> vlines = split(lines, ',');
    for (const std::string line: vlines)
    {
      std::istringstream line_stream(line);
      std::string objectName;
      std::cout << line << std::endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      double center[2] = {
        ffparser("-xpos").asDouble(.5*sim.extents[0]),
        ffparser("-ypos").asDouble(.5*sim.extents[1])
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
      else if (objectName=="smartDisk")
        shape = new SmartCylinder(    sim, ffparser, center);
      else if (objectName=="halfDisk")
        shape = new HalfDisk(         sim, ffparser, center);
      else if (objectName=="ellipse")
        shape = new Ellipse(          sim, ffparser, center);
      else if (objectName=="diskVarDensity")
        shape = new DiskVarDensity(   sim, ffparser, center);
      else if (objectName=="ellipseVarDensity")
        shape = new EllipseVarDensity(sim, ffparser, center);
      else if (objectName=="blowfish")
        shape = new BlowFish(         sim, ffparser, center);
      else if (objectName=="glider")
        shape = new Glider(           sim, ffparser, center);
      else if (objectName=="stefanfish")
        shape = new StefanFish(       sim, ffparser, center);
      else if (objectName=="carlingfish")
        shape = new CarlingFish(      sim, ffparser, center);
      else {
        std::cout << "FATAL - shape is not recognized!" << std::endl; abort();
      }
      assert(shape not_eq nullptr);
      shape->obstacleID = k++;
      sim.shapes.push_back(shape);
    }
  }

  if( sim.shapes.size() ==  0) {
    std::cout << "FATAL - Did not create any obstacles." << std::endl; abort();
  }

  //now that shapes are created, we know whether we need variable rho solver:
  sim.checkVariableDensity();
}

void Simulation::init()
{
  parseRuntime();
  createShapes();

  pipeline.clear();
  {
    IC ic(sim);
    ic(0);
  }
  if(sim.bVariableDensity)
  {
    pipeline.push_back( new PutObjectsOnGrid(sim) );
    pipeline.push_back( new advDiffGrav(sim) );
    pipeline.push_back( new FadeOut(sim) );
    if(sim.iterativePenalization)
      pipeline.push_back( new PressureVarRho_approx(sim) );
    else {
      pipeline.push_back( new PressureVarRho_proper(sim) );
      pipeline.push_back( new UpdateObjects(sim) );
    }
     // pipeline.push_back( new PressureVarRho_iterator(sim) );
    pipeline.push_back( new FadeOut(sim) );
  }
  else
  {
    pipeline.push_back( new PutObjectsOnGrid(sim) );
    pipeline.push_back( new advDiff(sim) );
    pipeline.push_back( new FadeOut(sim) );
    //pipeline.push_back( new PressureVarRho(sim) );
    //pipeline.push_back( new PressureVarRho_proper(sim) );
    if(sim.iterativePenalization)
      pipeline.push_back( new PressureIterator_unif(sim) );
    else {
      pipeline.push_back( new PressureSingle(sim) );
      pipeline.push_back( new UpdateObjects(sim) );
    }
    pipeline.push_back( new FadeOut(sim) );
  }

  std::cout << "Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    std::cout << "\t" << pipeline[c]->getName() << "\n";

  reset();
  sim.dumpAll("IC");
}

void Simulation::reset()
{
   sim.resetAll();
   IC ic(sim);
   ic(0);
   // put objects on grid
   (*pipeline[0])(0);
   ApplyObjVel initVel(sim);
   initVel(0);
}

void Simulation::simulate()
{
  while (1)
  {
    sim.startProfiler("DT");
    const double dt = calcMaxTimestep();
    sim.stopProfiler();
    if (advance(dt)) break;
  }
}

double Simulation::calcMaxTimestep()
{
  const auto findMaxU_op = findMaxU(sim);
  const Real maxU = findMaxU_op.run(); assert(maxU>=0);

  const double h = sim.getH();
  const double dtFourier = h*h/sim.nu, dtCFL = maxU<2.2e-16? 1 : h/maxU;
  const double maxUb = sim.maxRelSpeed(), dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  sim.dt = sim.CFL * std::min({dtCFL, dtFourier, dtBody});

  if (sim.step < 100)
  {
    const double x = (sim.step+1.0)/100;
    const double rampCFL = std::exp(std::log(1e-3)*(1-x) + std::log(sim.CFL)*x);
    sim.dt = rampCFL * std::min({dtCFL, dtFourier, dtBody});
  }

  if(sim.verbose)
    printf("step:%d, time:%f, dt=%f, uinf:[%f %f], maxU:%f\n",
      sim.step, sim.time, sim.dt, sim.uinfx, sim.uinfy, maxU);

  if(sim.dlm > 0) sim.lambda = sim.dlm / sim.dt;
  return sim.dt;
}

bool Simulation::advance(const double dt)
{
  assert(dt>2.2e-16);
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++) (*pipeline[c])(sim.dt);

  sim.time += sim.dt;
  sim.step++;

  //dump some time steps every now and then
  sim.startProfiler("Dump");
  if(bDump) {
    sim.registerDump();
    sim.dumpAll("avemaria_");
  }
  sim.stopProfiler();

  const bool bOver = sim.bOver();

  if (bOver || (sim.step % 50 == 0 && sim.verbose) ) sim.printResetProfiler();

  return bOver;
}
