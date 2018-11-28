//
//  CoordinatorPressure.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/30/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

#include "GenericCoordinator.h"

#include "HYPRE_struct_ls.h"

class OperatorDivergence : public GenericLabOperator
{
 private:
  const double dt;
  Real * const rhs_buffer;
  const size_t stride_buffer;

 public:
  OperatorDivergence(double DT, Real*const RHS, const size_t STRIDE):
    dt(DT), rhs_buffer(RHS), stride_buffer(STRIDE) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 4, 0,1,2,3);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }

  ~OperatorDivergence() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    // here i multiply the rhs by dt and obtain p*dt from poisson solver
    const Real factor = 0.5/(info.h_gridpoint); // should be [...]/dt
    const size_t blocki = BlockType::sizeX * info.index[0];
    const size_t blockj = BlockType::sizeY * info.index[1];
    const size_t blockStart = blocki + stride_buffer * blockj;
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement & phi  = lab(ix  , iy  );
      const FluidElement & phiN = lab(ix  , iy+1);
      const FluidElement & phiS = lab(ix  , iy-1);
      const FluidElement & phiE = lab(ix+1, iy  );
      const FluidElement & phiW = lab(ix-1, iy  );
      const Real divVel =           (phiE.u   -phiW.u    + phiN.v   -phiS.v);
      const Real divDef = phi.tmp * (phiE.tmpU-phiW.tmpU + phiN.tmpV-phiS.tmpV);
      rhs_buffer[blockStart + ix + stride_buffer*iy] = factor*(divVel-divDef);
    }
  }
};

class OperatorGradP : public GenericLabOperator
{
 private:
  const double dt;
 public:
  OperatorGradP(double _dt): dt(_dt) {
    stencil = StencilInfo(-1,-1,0, 2,2,1, false, 1, 5);
    stencil_start[0] = -1; stencil_start[1] = -1; stencil_start[2] = 0;
    stencil_end[0] = 2; stencil_end[1] = 2; stencil_end[2] = 1;
  }
  ~OperatorGradP() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const {
    const Real prefactor = -.5 / info.h_gridpoint; // * dt / dt
    const Real prescale = 1.0 / dt; // tmp is p * dt
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      // tmp contains the pressure correction after the Poisson solver
      o(ix,iy).u += prefactor * (lab(ix+1,iy).tmp - lab(ix-1,iy).tmp);
      o(ix,iy).v += prefactor * (lab(ix,iy+1).tmp - lab(ix,iy-1).tmp);
      o(ix,iy).p = o(ix,iy).tmp * prescale; // copy pressure onto field p
    }
  }
};

template <typename Lab>
class CoordinatorPressure : public GenericCoordinator
{
 protected:
  // total number of DOFs in y and x:
  const size_t totNy = sim.grid->getBlocksPerDimension(1) * _BS_;
  const size_t totNx = sim.grid->getBlocksPerDimension(0) * _BS_;
  // grid spacing:
  const Real h = sim.grid->getBlocksInfo().front().h_gridpoint;
  // memory buffer for mem transfers to/from hypre:
  Real * const buffer = new Real[totNy * totNx];
  int ilower[2] = {0,0};
  int iupper[2] = {(int)totNx-1, (int)totNy-1};
  const bool bPeriodic = false;
  const std::string solver = "pcg";
  HYPRE_StructGrid     hypre_grid;
  HYPRE_StructStencil  hypre_stencil;
  HYPRE_StructMatrix   hypre_mat;
  HYPRE_StructVector   hypre_rhs;
  HYPRE_StructVector   hypre_sol;
  HYPRE_StructSolver   hypre_solver;

  inline void getSolution() const
  {
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      const BlockInfo& info = vInfo[i];
      const size_t blocki = FluidGrid::BlockType::sizeX * info.index[0];
      const size_t blockj = FluidGrid::BlockType::sizeY * info.index[1];
      const size_t blockStart = blocki + totNx * blockj;
      FluidBlock& b = *(FluidBlock*)info.ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          b(ix,iy).tmp = buffer[blockStart + ix + totNx*iy];
    }
  }

  template <typename Operator>
  void computeSplit(const double dt, const Operator& K)
  {
    #pragma omp parallel
    {
      Lab mylab;
      mylab.prepare(*(sim.grid), K.stencil_start, K.stencil_end, false);
      #pragma omp for schedule(static)
      for(size_t i=0; i<vInfo.size(); i++) {
        const BlockInfo& info = vInfo[i];
        mylab.load(info, 0);
        K(mylab, info, *(FluidBlock*)info.ptrBlock);
      }
    }
  }

 public:
  void operator()(const double dt)
  {
    check("pressure - start");
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy).tmpU = 0; b(ix,iy).tmpV = 0;
      }
    }
    for( const auto& shape : sim.shapes ) shape->deformation_velocities();

    {
      const auto divKernel = OperatorDivergence(dt, buffer, totNx);
      computeSplit(dt, divKernel);

      HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer);

      if (solver == "gmres")
        HYPRE_StructGMRESSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
      else if (solver == "smg")
        HYPRE_StructSMGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
      else
        HYPRE_StructPCGSolve(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);

      HYPRE_StructVectorGetBoxValues(hypre_sol, ilower, iupper, buffer);
      getSolution();
      const auto gradPKernel = OperatorGradP(dt);
      computeSplit(dt, gradPKernel);
    }
    check("pressure - end");
  }

  CoordinatorPressure(SimulationData& s) : GenericCoordinator(s)
  {
    const auto COMM = MPI_COMM_SELF;
    // Grid
    HYPRE_StructGridCreate(COMM, 2, &hypre_grid);

    HYPRE_StructGridSetExtents(hypre_grid, ilower, iupper);

    if(bPeriodic)
    {
      // if grid is periodic, this function takes the period
      // length... ie. the grid size.
      HYPRE_StructGridSetPeriodic(hypre_grid, iupper);
    }

    HYPRE_StructGridAssemble(hypre_grid);

    { // Stencil
      int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};
      HYPRE_StructStencilCreate(2, 5, &hypre_stencil);
      for (int j = 0; j < 5; ++j)
        HYPRE_StructStencilSetElement(hypre_stencil, j, offsets[j]);
    }

    { // Matrix
      HYPRE_StructMatrixCreate(COMM, hypre_grid, hypre_stencil, &hypre_mat);
      HYPRE_StructMatrixInitialize(hypre_mat);

      // These indices must match to those in the offset array:
      int inds[5] = {0, 1, 2, 3, 4};
      using RowType = Real[5];
      RowType * vals = new RowType[totNy*totNx];

      #pragma omp parallel for schedule(static)
      for (size_t j = 0; j < totNy; j++)
      for (size_t i = 0; i < totNx; i++) {
        vals[j*totNx + i][0] = -4; // center
        vals[j*totNx + i][1] =  1; // west
        vals[j*totNx + i][2] =  1; // east
        vals[j*totNx + i][3] =  1; // south
        vals[j*totNx + i][4] =  1; // north
      }

      if(not bPeriodic) // 0 Neumann BC
      {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < totNx; i++) {
          vals[i][0] += 1; // center
          vals[i][3] -= 1; // south
          vals[totNx*(totNy-1) +i][0] += 1; // center
          vals[totNx*(totNy-1) +i][4] -= 1; // north
        }
        #pragma omp parallel for schedule(static)
        for (size_t j = 0; j < totNy; j++) {
          vals[totNx*j][0] += 1; // center
          vals[totNx*j][1] -= 0; // west
          vals[totNx*j + totNx-1][0] += 1; // center
          vals[totNx*j + totNx-1][2] -= 0; // east
        }
      }
      Real * const linV = (Real*) vals;
      HYPRE_StructMatrixSetBoxValues(hypre_mat, ilower, iupper, 5, inds, linV);
      delete [] vals;
      HYPRE_StructMatrixAssemble(hypre_mat);
    }


    // Rhs and initial guess
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_rhs);
    HYPRE_StructVectorCreate(COMM, hypre_grid, &hypre_sol);

    HYPRE_StructVectorInitialize(hypre_rhs);
    HYPRE_StructVectorInitialize(hypre_sol);

    {
      memset(buffer, 0, totNx*totNy*sizeof(Real));
      HYPRE_StructVectorSetBoxValues(hypre_rhs, ilower, iupper, buffer);
      HYPRE_StructVectorSetBoxValues(hypre_sol, ilower, iupper, buffer);
    }

    HYPRE_StructVectorAssemble(hypre_rhs);
    HYPRE_StructVectorAssemble(hypre_sol);

    if (solver == "gmres") // GMRES solver
    {
      HYPRE_StructGMRESCreate(COMM, &hypre_solver);
      HYPRE_StructGMRESSetTol(hypre_solver, 1e-6);
      HYPRE_StructGMRESSetPrintLevel(hypre_solver, 0);
      HYPRE_StructGMRESSetMaxIter(hypre_solver, 1000);
      HYPRE_StructGMRESSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
    }
    else
    if (solver == "smg") // SMG solver
    {
      HYPRE_StructSMGCreate(COMM, &hypre_solver);
      HYPRE_StructSMGSetMemoryUse(hypre_solver, 0);
      HYPRE_StructSMGSetMaxIter(hypre_solver, 1000);
      HYPRE_StructSMGSetTol(hypre_solver, 1e-6);
      HYPRE_StructSMGSetRelChange(hypre_solver, 0);
      HYPRE_StructSMGSetPrintLevel(hypre_solver, 0);
      HYPRE_StructSMGSetNumPreRelax(hypre_solver, 1);
      HYPRE_StructSMGSetNumPostRelax(hypre_solver, 1);

      HYPRE_StructSMGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
    }
    else // PCG solver
    {
      HYPRE_StructPCGCreate(COMM, &hypre_solver);
      HYPRE_StructPCGSetMaxIter(hypre_solver, 1000);
      HYPRE_StructPCGSetTol(hypre_solver, 1e-6);
      HYPRE_StructPCGSetPrintLevel(hypre_solver, 0);
      HYPRE_StructPCGSetup(hypre_solver, hypre_mat, hypre_rhs, hypre_sol);
    }
  }

  string getName() {
    return "hypre";
  }

  virtual ~CoordinatorPressure()
  {
    if (solver == "gmres")
      HYPRE_StructGMRESDestroy(hypre_solver);
    else if (solver == "smg")
      HYPRE_StructSMGDestroy(hypre_solver);
    else
      HYPRE_StructPCGDestroy(hypre_solver);
    HYPRE_StructGridDestroy(hypre_grid);
    HYPRE_StructStencilDestroy(hypre_stencil);
    HYPRE_StructMatrixDestroy(hypre_mat);
    HYPRE_StructVectorDestroy(hypre_rhs);
    HYPRE_StructVectorDestroy(hypre_sol);
    delete [] buffer;
  }
};
