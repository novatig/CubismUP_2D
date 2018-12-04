//
//  Sim_FSI_Gravity.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Simulation.h"

#include <HDF5Dumper.h>
//#include <ZBinDumper.h>

#include "Operators/Helpers.h"
#include "Operators/IC.h"
#include "Operators/PressureIterator.h"
#include "Operators/PutObjectsOnGrid.h"
#include "Operators/UpdateObjects.h"
#include "Operators/RKstep1.h"
#include "Operators/RKstep2.h"
#include "Utils/FactoryFileLineParser.h"

#include "Obstacles/ShapesSimple.h"
//#include "BlowFish.h"
//#include "StefanFish.h"
//#include "CarlingFish.h"

#include <regex>
#include <algorithm>
#include <iterator>

static inline vector<string> split(const string &s, const char delim) {
  stringstream ss(s); string item; vector<string> tokens;
  while (getline(ss, item, delim)) tokens.push_back(item);
  return tokens;
}

void Simulation::dump(string fname) {
  stringstream ss;
  ss<<"avemaria_"<<fname<<std::setfill('0')<<std::setw(7)<<sim.step;

  DumpHDF5<VectorGrid,StreamerVector>(*(sim.vel ), sim.step, sim.time,
    "vel_" + ss.str(), sim.path4serialization);
  //if(sim.bVariableDensity)
  DumpHDF5<ScalarGrid,StreamerScalar>(*(sim.pres), sim.step, sim.time,
    "pres_" + ss.str(), sim.path4serialization);
  //DumpZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
  //ReadZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
  cout << "=================================================================\n";
  cout << "\t\tFlow past a falling obstacle\n";
  cout << "=================================================================\n";
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
  cout << "bRestart is " << sim.bRestart << endl;

  // initialize grid
  parser.set_strict_mode();
  sim.bpdx = parser("-bpdx").asInt();
  sim.bpdy = parser("-bpdy").asInt();
  sim.allocateGrid();

  // simulation ending parameters
  parser.unset_strict_mode();
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // output parameters
  sim.dumpFreq = parser("-fdump").asInt(0);
  sim.dumpTime = parser("-tdump").asDouble(0);

  sim.path2file = parser("-file").asString("./");
  sim.path4serialization = parser("-serialization").asString(sim.path2file);
  if ( parser("-bFreeSpace").asInt(0) )
    sim.poissonType = 1;
  if ( parser("-bNeumann").asInt(0) )
    sim.poissonType = 2;

  // simulation settings
  sim.CFL = parser("-CFL").asDouble(.1);
  sim.lambda = parser("-lambda").asDouble(1e5);
  sim.dlm = parser("-dlm").asDouble(1);
  sim.nu = parser("-nu").asDouble(1e-2);

  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);//stronger silence, not even files
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes()
{
  parser.set_strict_mode();
  const Real ext = std::max(sim.bpdx, sim.bpdy);
  parser.unset_strict_mode();
  const std::string shapeArg = parser("-shapes").asString("");
  std::stringstream descriptors( shapeArg );
  string lines;
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
    for (const string line: vlines)
    {
      istringstream line_stream(line);
      string objectName;
      cout << line << endl;
      line_stream >> objectName;
      // Comments and empty lines ignored:
      if(objectName.empty() or objectName[0]=='#') continue;
      FactoryFileLineParser ffparser(line_stream);
      double center[2] = {
        ffparser("-xpos").asDouble(.5*sim.bpdx/ext),
        ffparser("-ypos").asDouble(.5*sim.bpdy/ext)
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
      //else if (objectName=="halfDisk")
      //  shape = new HalfDisk(         sim, ffparser, center);
      //else if (objectName=="ellipse")
      //  shape = new Ellipse(          sim, ffparser, center);
      //else if (objectName=="diskVarDensity")
      //  shape = new DiskVarDensity(   sim, ffparser, center);
      //else if (objectName=="ellipseVarDensity")
      //  shape = new EllipseVarDensity(sim, ffparser, center);
      //else if (objectName=="blowfish")
      //  shape = new BlowFish(         sim, ffparser, center);
      //else if (objectName=="stefanfish")
      //  shape = new StefanFish(       sim, ffparser, center);
      //else if (objectName=="carlingfish")
      //  shape = new CarlingFish(      sim, ffparser, center);
      else {
        cout << "FATAL - shape is not recognized!" << std::endl; abort();
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

  // setup initial conditions
  {
    IC ic(sim);
    sim.startProfiler(ic.getName());
    ic(0);
    sim.stopProfiler();
  }

  pipeline.clear();

  pipeline.push_back( new PutObjectsOnGrid(sim) );
  pipeline.push_back( new RKstep1(sim) );
  pipeline.push_back( new RKstep2(sim) );
  pipeline.push_back( new PressureIterator(sim) );
  pipeline.push_back( new UpdateObjects(sim) );

  cout << "Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    cout << "\t" << pipeline[c]->getName() << endl;
}

void Simulation::simulate() {
  while (1) {
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
  sim.dt = sim.CFL * std::min(dtCFL, dtFourier);

  const double maxUb = sim.maxRelSpeed();
  const double dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  sim.dt = std::min( sim.dt, sim.CFL*dtBody );

  if(sim.dlm >= 1) sim.lambda = sim.dlm / sim.dt;
  if (sim.step < 100) {
    const double x = (sim.step+1.0)/100;
    const double rampCFL = std::exp(std::log(1e-3)*(1-x) + std::log(sim.CFL)*x);
    sim.dt = rampCFL*std::min({dtCFL, dtFourier, dtBody});
  }
  #ifndef RL_TRAIN
  if(sim.verbose)
    cout << "step, time, dt "// (Fourier, CFL, body): "
    <<sim.step<<" "<<sim.time<<" "<<sim.dt<<" "<<sim.uinfx<<" "<<sim.uinfy
    //<<" "<<dtFourier<<" "<<dtCFL<<" "<<dtBody
    <<endl;
  #endif

  return sim.dt;
}

bool Simulation::advance(const double dt)
{
  assert(dt>2.2e-16);
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++)
  {
    sim.startProfiler(pipeline[c]->getName());
    (*pipeline[c])(sim.dt);
    sim.stopProfiler();
    // stringstream ss; ss<<path2file<<"avemaria_"<<pipeline[c]->getName();
    // ss<<"_"<<std::setfill('0')<<std::setw(7)<<step<<".vti"; dump(ss);
  }

  sim.time += sim.dt;
  sim.step++;

  //dump some time steps every now and then
  sim.startProfiler("Dump");
  if(bDump) {
    sim.registerDump();
    dump();
  }
  sim.stopProfiler();

  const bool bOver = sim.bOver();

  if (bOver || (sim.step % 5 == 0 && sim.verbose) ) sim.printResetProfiler();

  return bOver;
}
