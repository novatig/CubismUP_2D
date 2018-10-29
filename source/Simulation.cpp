//
//  Sim_FSI_Gravity.cpp
//  CubismUP_2D
//
//  Created by Christian Conti on 1/26/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#include "Simulation.h"

#ifdef USE_VTK
#include <SerializerIO_ImageVTK.h>
#elif defined(_USE_HDF_)
#include <HDF5Dumper.h>
#endif
//#include <ZBinDumper.h>

#include "HelperOperators.h"
#include "CoordinatorIC.h"
#include "CoordinatorAdvection.h"
#include "CoordinatorMultistep.h"
#include "CoordinatorDiffusion.h"
#include "CoordinatorShape.h"
#include "CoordinatorPressure.h"
#include "CoordinatorGravity.h"
#include "FactoryFileLineParser.h"

#include "ShapesSimple.h"
#include "BlowFish.h"
#include "StefanFish.h"
#include "CarlingFish.h"
#include "Profiler.h"

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
  #ifdef USE_VTK
    SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;
    dumper.Write( *(sim.grid), sim.path4serialization + ss.str() + ".vti" );
  #elif defined(_USE_HDF_)
    DumpHDF5<FluidGrid,StreamerVelocityVector>(*(sim.grid), sim.step, sim.time,
      StreamerVelocityVector::prefix() + ss.str(), sim.path4serialization);
    //if(sim.bVariableDensity)
      DumpHDF5<FluidGrid,StreamerPressure>(*(sim.grid), sim.step, sim.time,
      StreamerPressure::prefix() + ss.str(), sim.path4serialization);
    if(sim.bVariableDensity)
      DumpHDF5<FluidGrid,StreamerRho>(*(sim.grid), sim.step, sim.time,
      StreamerRho::prefix() + ss.str(), sim.path4serialization);
  #endif
  //DumpZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
  //ReadZBin<FluidGrid, StreamerSerialization>(*grid, serializedGrid.str(), path4serialization);
}

Simulation::Simulation(int argc, char ** argv) : parser(argc,argv)
{
  #ifndef SMARTIES_APP
    profiler = new Profiler();
  #endif
  cout << "=================================================================\n";
  cout << "\t\tFlow past a falling obstacle\n";
  cout << "=================================================================\n";
}

Simulation::~Simulation() {
  #ifndef SMARTIES_APP
    delete profiler;
  #endif
  while( not pipeline.empty() ) {
    GenericCoordinator * g = pipeline.back();
    pipeline.pop_back();
    if(g not_eq nullptr) delete g;
  }
}

void Simulation::parseRuntime() {
  sim.bRestart = parser("-restart").asBool(false);
  cout << "bRestart is " << sim.bRestart << endl;

  // initialize grid
  parser.set_strict_mode();
  const int bpdx = parser("-bpdx").asInt();
  const int bpdy = parser("-bpdy").asInt();
  sim.grid = new FluidGrid(bpdx, bpdy, 1);
  assert( sim.grid not_eq nullptr );

  // simulation ending parameters
  parser.unset_strict_mode();
  sim.nsteps = parser("-nsteps").asInt(0);
  sim.endTime = parser("-tend").asDouble(0);

  // output parameters
  sim.dumpFreq = parser("-fdump").asDouble(0);
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
  sim.dlm = parser("-dlm").asDouble(10.);
  sim.nu = parser("-nu").asDouble(1e-2);

  sim.verbose = parser("-verbose").asInt(1);
  sim.muteAll = parser("-muteAll").asInt(0);//stronger silence, not even files
  if(sim.muteAll) sim.verbose = 0;
}

void Simulation::createShapes() {
  parser.set_strict_mode();
  const Real axX = parser("-bpdx").asInt();
  const Real axY = parser("-bpdy").asInt();
  const Real ext = std::max(axX, axY);
  parser.unset_strict_mode();
  const string shapeArg = parser("-shapes").asString("");
  stringstream descriptors( shapeArg );
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
    const vector<string> vlines = split(lines, ',');
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
        ffparser("-xpos").asDouble(.5*axX/ext),
        ffparser("-ypos").asDouble(.5*axY/ext)
      };
      //ffparser.print_args();
      Shape* shape = nullptr;
      if (objectName=="disk")
        shape = new Disk(             sim, ffparser, center);
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
      else if (objectName=="stefanfish")
        shape = new StefanFish(       sim, ffparser, center);
      else if (objectName=="carlingfish")
        shape = new CarlingFish(      sim, ffparser, center);
      else {
        cout << "Error - this shape is not recognized! Aborting now\n";
        abort();
      }
      assert(shape not_eq nullptr);
      shape->obstacleID = k++;
      sim.shapes.push_back(shape);
    }
  }

  if( sim.shapes.size() ==  0) {
    std::cout << "Did not create any obstacles. Not supported. Aborting!\n";
    abort();
  }

  //now that shapes are created, we know whether we need variable rho solver:
  sim.checkVariableDensity();
}

void Simulation::init() {
  parseRuntime();
  createShapes();

  // setup initial conditions
  CoordinatorIC coordIC(sim);
  #ifndef SMARTIES_APP
    profiler->push_start(coordIC.getName());
  #endif
  coordIC(0);
  #ifndef SMARTIES_APP
    profiler->pop_stop();
  #endif

  pipeline.clear();

  pipeline.push_back( new CoordinatorComputeShape(sim) );
  pipeline.push_back( new CoordinatorVelocities(sim) );
  pipeline.push_back( new CoordinatorPenalization(sim) );

  #if 1 // in one sweep advect, diffuse, add hydrostatic
    pipeline.push_back( new CoordinatorMultistep<Lab>(sim) );
  #else
    pipeline.push_back( new CoordinatorAdvection<Lab>(sim) );
    pipeline.push_back( new CoordinatorDiffusion<Lab>(sim) );
    pipeline.push_back( new CoordinatorGravity(sim) );
  #endif

  pipeline.push_back( new CoordinatorPressure<Lab>(sim) );
  pipeline.push_back( new CoordinatorComputeForces(sim) );
  if( sim.poissonType not_eq 1 )
    pipeline.push_back( new CoordinatorFadeOut(sim) );

  cout << "Coordinator/Operator ordering:\n";
  for (size_t c=0; c<pipeline.size(); c++)
    cout << "\t" << pipeline[c]->getName() << endl;

  (*pipeline[0])(0);
  dump("init");
}

void Simulation::simulate() {
  while (1) {
    #ifndef SMARTIES_APP
     profiler->push_start("DT");
    #endif
    const double dt = calcMaxTimestep();
    #ifndef SMARTIES_APP
     profiler->pop_stop();
    #endif
    if (advance(dt)) break;
  }
}

double Simulation::calcMaxTimestep() {
  const Real maxU = findMaxUOMP( sim ); assert(maxU>=0);
  const double h = sim.getH();
  const double dtFourier = h*h/sim.nu, dtCFL = maxU<2.2e-16? 1 : h/maxU;
  sim.dt = sim.CFL * std::min(dtCFL, dtFourier);

  const double maxUb = sim.maxRelSpeed();
  const double dtBody = maxUb<2.2e-16? 1 : h/maxUb;
  sim.dt = std::min( sim.dt, sim.CFL*dtBody );

  if(sim.dlm > 1) sim.lambda = sim.dlm / sim.dt;
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

bool Simulation::advance(const double dt) {
  assert(dt>2.2e-16);
  const bool bDump = sim.bDump();

  for (size_t c=0; c<pipeline.size(); c++) {
    #ifndef SMARTIES_APP
     profiler->push_start(pipeline[c]->getName());
    #endif
    (*pipeline[c])(sim.dt);
    #ifndef SMARTIES_APP
     profiler->pop_stop();
    #endif
    // stringstream ss; ss<<path2file<<"avemaria_"<<pipeline[c]->getName();
    // ss<<"_"<<std::setfill('0')<<std::setw(7)<<step<<".vti"; dump(ss);
  }

  sim.time += sim.dt;
  sim.step++;

  //dump some time steps every now and then
  #ifndef SMARTIES_APP
   profiler->push_start("Dump");
  #endif
  if(bDump) {
    sim.registerDump();
    dump();
  }
  #ifndef SMARTIES_APP
   profiler->pop_stop();
  #endif

  if (sim.step % 100 == 0 && sim.verbose) {
    #ifndef SMARTIES_APP
     profiler->printSummary();
     profiler->reset();
    #endif
  }

  const bool bOver = sim.bOver();
  #ifndef SMARTIES_APP
   if(bOver) profiler->printSummary();
  #endif
  return bOver;
}
