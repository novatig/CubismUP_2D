//
//  IF2D_ObstacleOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 04/10/14.
//
//


#ifndef IF2D_ROCKS_IF2D_ObstacleOperator_h
#define IF2D_ROCKS_IF2D_ObstacleOperator_h

#include "common.h"
#include "Definitions.h"
//#include "IF2D_ObstacleLibrary.h"
#include "IF2D_FactoryFileLineParser.h"
#include <fstream>

// forward declaration of derived class for visitor

class IF2D_ObstacleOperator;
class IF2D_ObstacleVector;

struct ObstacleVisitor
{
    virtual void visit(IF2D_ObstacleOperator * obstacle) {}
    virtual void visit(IF2D_ObstacleVector * obstacle) {}
};

class IF2D_ObstacleOperator
{
protected:
    FluidGrid * grid;
    surfacePoints surfData;
    vector<BlockInfo> vInfo;
    std::map<int,ObstacleBlock*> obstacleBlocks;
    
    Real position[2], transVel[2], angVel, angle, ext_pos[2], area, J; // moment of inertia
    Real mass, force[2], torque; //from diagnostics
    Real totChi, totFx, totFy, drag, thrust, Pout, PoutBnd, defPower, defPowerBnd, Pthrust, Pdrag, EffPDef, EffPDefBnd; //from compute forces
    double transVel_correction[2], angVel_correction;
    double *pX, *pY;
    int Npts;


    virtual void _parseArguments(ArgumentParser & parser);

    virtual void _writeComputedVelToFile(const int step_id, const Real t);

    virtual void _writeDiagForcesToFile(const int step_id, const Real t);

    virtual void _makeDefVelocitiesMomentumFree(const double CoM[2]);

public:
    int obstacleID;
    IF2D_ObstacleOperator(FluidGrid * grid, ArgumentParser& parser) :
    	grid(grid), obstacleID(0), transVel{0.0,0.0}, angVel(0.0), area(0.0), J(0.0), pY(nullptr), pX(nullptr), Npts(0)
    {
        vInfo = grid->getBlocksInfo();
        _parseArguments(parser);
        ext_pos[0]=position[0];
        ext_pos[1]=position[1];
    }

    IF2D_ObstacleOperator(FluidGrid * grid):
    grid(grid), obstacleID(0), transVel{0.0,0.0}, angVel(0.0), area(0.0), J(0.0), pY(nullptr), pX(nullptr), Npts(0)
	{
    	vInfo = grid->getBlocksInfo();
	}
    
    void Accept(ObstacleVisitor * visitor)
    {
    	visitor->visit(this);
    }

    virtual void computeDiagnostics(const int stepID, const double time, const double* Uinf, const double lambda) ;
    virtual void computeVelocities(const double* Uinf);
    virtual void computeForces(const int stepID, const double time, const double* Uinf, const double NU, const bool bDump);
    virtual void update(const int step_id, const double t, const double dt);
    virtual void save(const int step_id, const double t, std::string filename = std::string());
    virtual void restart(const double t, std::string filename = std::string());
    
    // some non-pure methods
    virtual void create(const int step_id,const double time, const double dt) { }
    
    //methods that work for all obstacles
    const std::map<int,ObstacleBlock*> getObstacleBlocks() const
    {
        return obstacleBlocks;
    }

    void getObstacleBlocks(std::map<int,ObstacleBlock*>*& obstblock_ptr)
    {
        obstblock_ptr = &obstacleBlocks;
    }
    
    virtual void characteristic_function();

    virtual std::vector<int> intersectingBlockIDs(const int buffer);

    virtual ~IF2D_ObstacleOperator()
    {
        for(auto & entry : obstacleBlocks) {
            if(entry.second != nullptr) {
                delete entry.second;
                entry.second = nullptr;
            }
        }
        obstacleBlocks.clear();
    }
    
    virtual void getTranslationVelocity(Real UT[2]) const;
    virtual void getAngularVelocity(Real & W) const;
    virtual void getCenterOfMass(Real CM[2]) const;
    virtual void setTranslationVelocity(Real UT[2]);
    virtual void setAngularVelocity(const Real W);
    double getForceX() const;
    double getForceY() const;
};

#endif
