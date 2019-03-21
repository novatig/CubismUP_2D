//
//  CubismUP_2D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#pragma once

#include "Definitions.h"
#include "Profiler.h"

class Shape;

struct SimulationData
{
  #ifndef SMARTIES_APP
    Profiler * profiler = new Profiler();
  #endif

  ScalarGrid * chi   = nullptr;
  VectorGrid * vel   = nullptr;
  VectorGrid * uDef  = nullptr;
  ScalarGrid * pres  = nullptr;
  VectorGrid * force = nullptr;

  ScalarGrid * pRHS  = nullptr;
  VectorGrid * tmpV  = nullptr;
  VectorGrid * DdF   = nullptr;
  ScalarGrid * tmp   = nullptr;
  ScalarGrid * invRho= nullptr;
  ScalarGrid * pOld  = nullptr;
  DumpGrid   * dump  = nullptr;

  void allocateGrid();

  std::vector<Shape*> shapes;

  double time = 0;
  int step = 0;

  Real uinfx = 0;
  Real uinfy = 0;

  int bpdx = 0;
  int bpdy = 0;

  double lambda = 0;
  double nu = 0;
  double dlm = 1;

  std::array<Real,2> gravity = { (Real) 0.0, (Real) -9.80665 };
  // nsteps==0 means that this stopping criteria is not active
  int nsteps = 0;
  // endTime==0  means that this stopping criteria is not active
  double endTime = 0;

  double dt = 0;
  double CFL = 0.1;

  bool verbose = true;
  bool muteAll = false;
  std::string poissonType = "hypre";
  bool bVariableDensity = false;
  // output
  // dumpFreq==0 means that this dumping frequency (in #steps) is not active
  int dumpFreq = 0;
  // dumpTime==0 means that this dumping frequency (in time)   is not active
  double dumpTime = 0;
  double nextDumpTime = 0;
  bool _bDump = false;
  bool bPing = false;
  bool bRestart = false;

  std::string path4serialization;
  std::string path2file;

  void resetAll();
  bool bDump();
  void registerDump();
  bool bOver() const;

  double minRho() const;
  double maxSpeed() const;
  double maxRelSpeed() const;
  void checkVariableDensity();

  inline double getH() const
  {
    return vel->getBlocksInfo().front().h_gridpoint; // yikes
  }

  void startProfiler(std::string name);
  void stopProfiler();
  void printResetProfiler();
  ~SimulationData();

  void dumpChi   (std::string name);
  void dumpPres  (std::string name);
  void dumpPrhs  (std::string name);
  void dumpTmp   (std::string name);
  void dumpVel   (std::string name);
  void dumpUobj  (std::string name);
  void dumpForce (std::string name);
  void dumpTmpV  (std::string name);
  void dumpAll   (std::string name);
  void dumpInvRho(std::string name);
  void dumpGlue  (std::string name);
};