//
//  OperatorDiffusion.h
//  CubismUP_2D
//
//	Operates on
//		tmpU, tmpV
//
//  Created by Christian Conti on 1/7/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_OperatorDiffusion_h
#define CubismUP_2D_OperatorDiffusion_h

#include "GenericOperator.h"

class OperatorDiffusion : public GenericLabOperator
{
private:
	const double mu;
	double dt;
	const int stage;
	
public:
	OperatorDiffusion(double dt, double mu, const int stage) : mu(mu), dt(dt), stage(stage)
	{
		stencil_start[0] = -1;
		stencil_start[1] = -1;
		stencil_start[2] = 0;
		stencil_end[0] = 2;
		stencil_end[1] = 2;
		stencil_end[2] = 1;
	}
	
	~OperatorDiffusion() {}
	
	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
	{
		const double prefactor = mu * dt / (info.h_gridpoint*info.h_gridpoint);
		
		// stage 1 of RK2
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
		for(int ix=0; ix<FluidBlock::sizeX; ++ix)
		{
			FluidElement& phi  = lab(ix,iy);
			FluidElement& phiN = lab(ix,iy+1);
			FluidElement& phiS = lab(ix,iy-1);
			FluidElement& phiE = lab(ix+1,iy);
			FluidElement& phiW = lab(ix-1,iy);
			const Real fac =prefactor/phi.rho;
			o(ix,iy).tmpU = phi.u + fac * (phiN.u + phiS.u + phiE.u + phiW.u - phi.u*4.);
			o(ix,iy).tmpV = phi.v + fac * (phiN.v + phiS.v + phiE.v + phiW.v - phi.v*4.);

#ifdef _MULTIPHASE_
			o(ix,iy).tmp = phi.rho + fac *(phiN.rho + phiS.rho + phiE.rho + phiW.rho - phi.rho*4.);
#endif
		}
   }
};

class OperatorDiffusionHighOrder : public GenericLabOperator
{
private:
	double mu;
	double dt;
	const int stage;
	
public:
	OperatorDiffusionHighOrder(double dt, double mu, const int stage) : mu(mu), dt(dt), stage(stage)
	{
		// it might be working - but the error is too small to measure!
		cout << "This operator needs debugging - it's not working correctly!\n";
		abort();
		
		stencil_start[0] = -2;
		stencil_start[1] = -2;
		stencil_start[2] = 0;
		stencil_end[0] = 3;
		stencil_end[1] = 3;
		stencil_end[2] = 1;
        
#ifdef _RK2_
        dt = (stage==0) ? dt*.5 : dt;
#endif
	}
	
	~OperatorDiffusionHighOrder() {}
	
	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
	{
		const double prefactor = mu * dt / (12*info.h_gridpoint*info.h_gridpoint);
		
		if (stage==0)
			for(int iy=0; iy<FluidBlock::sizeY; ++iy)
				for(int ix=0; ix<FluidBlock::sizeX; ++ix)
				{
					FluidElement& phi = lab(ix,iy);
					FluidElement& phiW = lab(ix-1,iy);
					FluidElement& phiE = lab(ix+1,iy);
					FluidElement& phiS = lab(ix,iy-1);
					FluidElement& phiN = lab(ix,iy+1);
					FluidElement& phiW2 = lab(ix-2,iy);
					FluidElement& phiE2 = lab(ix+2,iy);
					FluidElement& phiS2 = lab(ix,iy-2);
					FluidElement& phiN2 = lab(ix,iy+2);
                    
#ifdef _DENSITYDIFF_
                    o(ix,iy).tmp = phi.rho + 1e-5 * dt / (info.h_gridpoint*info.h_gridpoint) * (-(phiN2.rho + phiS2.rho + phiE2.rho + phiW2.rho) + 16*(phiN.rho + phiS.rho + phiE.rho + phiW.rho) - 60.*phi.rho);
#endif
                    
#ifdef _MULTIPHASE_
					o(ix,iy).tmpU += phi.u + prefactor/phi.rho * (-(phiN2.u + phiS2.u + phiE2.u + phiW2.u) + 16*(phiN.u + phiS.u + phiE.u + phiW.u) - 60.*phi.u);
                    o(ix,iy).tmpV += phi.v + prefactor/phi.rho * (-(phiN2.v + phiS2.v + phiE2.v + phiW2.v) + 16*(phiN.v + phiS.v + phiE.v + phiW.v) - 60.*phi.v);
#else
#ifdef _CONSTNU_
                    o(ix,iy).tmpU += phi.u + prefactor * (-(phiN2.u + phiS2.u + phiE2.u + phiW2.u) + 16*(phiN.u + phiS.u + phiE.u + phiW.u) - 60.*phi.u);
                    o(ix,iy).tmpV += phi.v + prefactor * (-(phiN2.v + phiS2.v + phiE2.v + phiW2.v) + 16*(phiN.v + phiS.v + phiE.v + phiW.v) - 60.*phi.v);
#else
                    o(ix,iy).tmpU += phi.u + prefactor/phi.rho * (-(phiN2.u + phiS2.u + phiE2.u + phiW2.u) + 16*(phiN.u + phiS.u + phiE.u + phiW.u) - 60.*phi.u);
                    o(ix,iy).tmpV += phi.v + prefactor/phi.rho * (-(phiN2.v + phiS2.v + phiE2.v + phiW2.v) + 16*(phiN.v + phiS.v + phiE.v + phiW.v) - 60.*phi.v);
#endif // _CONSTNU_
#endif // _MULTIPHASE_
                }
#ifdef _RK2_
        else if (stage==1)
            for(int iy=0; iy<FluidBlock::sizeY; ++iy)
                for(int ix=0; ix<FluidBlock::sizeX; ++ix)
                {
                    FluidElement& phi = lab(ix,iy);
                    FluidElement& phiN = lab(ix,iy+1);
                    FluidElement& phiS = lab(ix,iy-1);
                    FluidElement& phiE = lab(ix+1,iy);
                    FluidElement& phiW = lab(ix-1,iy);
                    FluidElement& phiN2 = lab(ix,iy+2);
                    FluidElement& phiS2 = lab(ix,iy-2);
                    FluidElement& phiE2 = lab(ix+2,iy);
                    FluidElement& phiW2 = lab(ix-2,iy);
            
#ifdef _DENSITYDIFF_
                    o(ix,iy).tmp = phi.rho + 1e-5 * dt / (info.h_gridpoint*info.h_gridpoint) * (-(phiN2.tmp + phiS2.tmp + phiE2.tmp + phiW2.tmp) + 16*(phiN.tmp + phiS.tmp + phiE.tmp + phiW.tmp) - 60.*phi.tmp);
#endif
            
#ifdef _MULTIPHASE_
                    o(ix,iy).tmpU += phi.u + prefactor/phi.rho * (-(phiN2.tmpU + phiS2.tmpU + phiE2.tmpU + phiW2.tmpU) + 16*(phiN.tmpU + phiS.tmpU + phiE.tmpU + phiW.tmpU) - 60.*phi.tmpU);
                    o(ix,iy).tmpV += phi.v + prefactor/phi.rho * (-(phiN2.tmpV + phiS2.tmpV + phiE2.tmpV + phiW2.tmpV) + 16*(phiN.tmpV + phiS.tmpV + phiE.tmpV + phiW.tmpV) - 60.*phi.tmpV);
#else
#ifdef _CONSTNU_
                    o(ix,iy).tmpU += phi.u + prefactor * (-(phiN2.tmpU + phiS2.tmpU + phiE2.tmpU + phiW2.tmpU) + 16*(phiN.tmpU + phiS.tmpU + phiE.tmpU + phiW.tmpU) - 60.*phi.tmpU);
                    o(ix,iy).tmpV += phi.v + prefactor * (-(phiN2.tmpV + phiS2.tmpV + phiE2.tmpV + phiW2.tmpV) + 16*(phiN.tmpV + phiS.tmpV + phiE.tmpV + phiW.tmpV) - 60.*phi.tmpV);
#else
                    o(ix,iy).tmpU += phi.u + prefactor/phi.rho * (-(phiN2.tmpU + phiS2.tmpU + phiE2.tmpU + phiW2.tmpU) + 16*(phiN.tmpU + phiS.tmpU + phiE.tmpU + phiW.tmpU) - 60.*phi.tmpU);
                    o(ix,iy).tmpV += phi.v + prefactor/phi.rho * (-(phiN2.tmpV + phiS2.tmpV + phiE2.tmpV + phiW2.tmpV) + 16*(phiN.tmpV + phiS.tmpV + phiE.tmpV + phiW.tmpV) - 60.*phi.tmpV);
#endif // _CONSTNU_
#endif // _MULTIPHASE_
                }
#endif // _RK2_
	}
};



class OperatorLaplace : public GenericLabOperator
{
private:
    double dt;
    
public:
    OperatorLaplace(double dt) : dt(dt)
    {
        /*
        stencil_start[0] = -2;
        stencil_start[1] = -2;
        stencil_start[2] = 0;
        stencil_end[0] = 3;
        stencil_end[1] = 3;
        stencil_end[2] = 1;
        /*/
        stencil_start[0] = -1;
        stencil_start[1] = -1;
        stencil_start[2] = 0;
        stencil_end[0] = 2;
        stencil_end[1] = 2;
        stencil_end[2] = 1;
        //*/
    }
    
    ~OperatorLaplace() {}
    
    template <typename Lab, typename BlockType>
    void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
    {
        //const double prefactor = 1. / (12*info.h_gridpoint*info.h_gridpoint);
        const double prefactor = 1. / (info.h_gridpoint*info.h_gridpoint);
        
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
            for(int ix=0; ix<FluidBlock::sizeX; ++ix)
            {
                FluidElement& phi = lab(ix,iy);
                FluidElement& phiN = lab(ix,iy+1);
                FluidElement& phiS = lab(ix,iy-1);
                FluidElement& phiE = lab(ix+1,iy);
                FluidElement& phiW = lab(ix-1,iy);
                /*
                FluidElement& phiN2 = lab(ix,iy+2);
                FluidElement& phiS2 = lab(ix,iy-2);
                FluidElement& phiE2 = lab(ix+2,iy);
                FluidElement& phiW2 = lab(ix-2,iy);
                */
                //o(ix,iy).v = prefactor * (-(phiN2.divU + phiS2.divU + phiE2.divU + phiW2.divU) + 16*(phiN.divU + phiS.divU + phiE.divU + phiW.divU) - 60.*phi.divU);
                o(ix,iy).v = prefactor * (phiN.divU + phiS.divU + phiE.divU + phiW.divU - 4.*phi.divU);
            }
    }
};

class OperatorViscousDrag : public GenericLabOperator
{
private:
	double dt;
	Real viscousDrag;
	
public:
	OperatorViscousDrag(double dt) : dt(dt), viscousDrag(0)
	{
		stencil_start[0] = -1;
		stencil_start[1] = -1;
		stencil_start[2] = 0;
		stencil_end[0] = 2;
		stencil_end[1] = 2;
		stencil_end[2] = 1;
	}
	
	~OperatorViscousDrag() {}
	
	template <typename Lab, typename BlockType>
	void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
	{
		const double prefactor = 1. / (info.h_gridpoint*info.h_gridpoint);
		viscousDrag = 0;
		
		for(int iy=0; iy<FluidBlock::sizeY; ++iy)
		for(int ix=0; ix<FluidBlock::sizeX; ++ix)
		{
			FluidElement& phi = lab(ix,iy);
			FluidElement& phiN = lab(ix,iy+1);
			FluidElement& phiS = lab(ix,iy-1);
			FluidElement& phiE = lab(ix+1,iy);
			FluidElement& phiW = lab(ix-1,iy);
			
			viscousDrag += prefactor * (phiN.divU + phiS.divU + phiE.divU + phiW.divU - 4.*phi.divU);
		}
	}
	
	inline Real getDrag()
	{
		return viscousDrag;
	}
};

#endif
