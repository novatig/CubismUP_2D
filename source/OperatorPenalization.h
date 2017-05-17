//
//  OperatorPenalization.h
//  CubismUP_2D
//
//	Operates on
//		u, v
//
//  Created by Christian Conti on 1/7/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_OperatorPenalization_h
#define CubismUP_2D_OperatorPenalization_h

#include "GenericOperator.h"

class OperatorPenalization : public GenericOperator
{
private:
    const double dt;
	const Real uBody[2];
	const Real omegaBody;
	const Real centerOfMass[2];
	const double lambda;
	
public:
	OperatorPenalization(double dt, Real uSolid, Real vSolid, Real omegaBody, Real xCenterOfMass, Real yCenterOfMass, double lambda) : dt(dt), uBody{uSolid,vSolid}, omegaBody(omegaBody), centerOfMass{xCenterOfMass,yCenterOfMass}, lambda(lambda) {}
    ~OperatorPenalization() {}
    
    void operator()(const BlockInfo& info, FluidBlock& block) const
    {
		// this implementation considers that the Euler updates has already happened
		// do we need a finite state machine coordinating operators?
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
            for(int ix=0; ix<FluidBlock::sizeX; ++ix)
			{
				Real p[2] = {0,0};
                info.pos(p,ix,iy);
                p[0] -= centerOfMass[0];
                p[1] -= centerOfMass[1];
                const Real alpha = 1./(1. +dt * lambda * block(ix,iy).chi);

#ifndef _MOVING_FRAME_
                const Real U_TOT[2] = {
                		uBody[0] - omegaBody*p[1],
						uBody[1] + omegaBody*p[0]
				};
#else
                const Real U_TOT[2] = {
                		- omegaBody*p[1],
						+ omegaBody*p[0]
				};
#endif

                block(ix,iy).u = alpha*block(ix,iy).u + (1-alpha)*U_TOT[0];
                block(ix,iy).v = alpha*block(ix,iy).v + (1-alpha)*U_TOT[1];

			}
    }
};


#endif
