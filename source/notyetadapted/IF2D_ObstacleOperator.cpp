//
//  IF2D_MovingObstacleOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 06/10/14.
//
//

#include "IF2D_ObstacleOperator.h"

void IF2D_ObstacleOperator::_makeDefVelocitiesMomentumFree(const double CoM[2])
{
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();

    double _A(0.0), _J(0.0), _momX(0.0), _momY(0.0), _momAng(0.0);
#pragma omp parallel for schedule(static) reduction(+:_A,_J,_momX,_momY,_momAng)
    for(int i=0; i<vInfo.size(); i++) {
        std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);
        if(pos == obstacleBlocks.end()) continue;
        BlockInfo info = vInfo[i];

        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            const double Xs = pos->second->chi[iy][ix];
            if (Xs == 0) continue;
            double p[2];
            info.pos(p, ix, iy);
            p[0]-=CoM[0];
            p[1]-=CoM[1];
            const double uDef = pos->second->udef[iy][ix][0];
            const double vDef = pos->second->udef[iy][ix][1];
            _A      += Xs;
            _momX   += Xs * uDef;
            _momY   += Xs * vDef;
            _momAng += Xs * (p[0]*vDef - p[1]*uDef);
            _J      += Xs * (p[0]*p[0] + p[1]*p[1]);
        }
    }
    //to be an MPI reduction:
    transVel_correction[0] = _momX/_A;
    transVel_correction[1] = _momY/_A;
    angVel_correction = _momAng/_J;

#pragma omp parallel for schedule(static)
    for(int i=0; i<vInfo.size(); i++) {
        std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);
        if(pos == obstacleBlocks.end()) continue;
        BlockInfo info = vInfo[i];
        FluidBlock& b = *(FluidBlock*)ary[i].ptrBlock;
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            double p[2];
            info.pos(p, ix, iy);
            p[0]-=CoM[0];
            p[1]-=CoM[1];
            const double correctVel[2] = {
                transVel_correction[0] - angVel_correction*p[1],
                transVel_correction[1] + angVel_correction*p[0]
            };
            pos->second->udef[iy][ix][0] -= correctVel[0];
            pos->second->udef[iy][ix][1] -= correctVel[1];

            //b(ix,iy).u = pos->second->udef[iy][ix][0];
            //b(ix,iy).v = pos->second->udef[iy][ix][1];
            //b(ix,iy).chi = pos->second->chi[iy][ix];
        }
    }

#ifndef NDEBUG
    {
        double _Af(0.0), _Jf(0.0), _momXf(0.0), _momYf(0.0), _momAngf(0.0);
#pragma omp parallel for schedule(static) reduction(+:_momXf,_momYf,_momAngf)
        for(int i=0; i<vInfo.size(); i++) {
            std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);
            if(pos == obstacleBlocks.end()) continue;
            BlockInfo info = vInfo[i];

            for(int iy=0; iy<FluidBlock::sizeY; ++iy)
            for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
                const double Xs = pos->second->chi[iy][ix];
                if (Xs == 0) continue;
                double p[2];
                info.pos(p, ix, iy);
                p[0]-=CoM[0];
                p[1]-=CoM[1];
                const double uDef = pos->second->udef[iy][ix][0];
                const double vDef = pos->second->udef[iy][ix][1];
                //_Af      += Xs;
                _momXf   += Xs * uDef;
                _momYf   += Xs * vDef;
                _momAngf += Xs * (p[0]*uDef - p[1]*vDef);
                //_Jf      += Xs * (p[0]*p[0] + p[1]*p[1]);
            }
        }
        printf("x linear momentum after correction of %10.10e: %10.10e\n",_momX,_momXf);
        printf("y linear momentum after correction of %10.10e: %10.10e\n",_momY,_momYf);
        printf(" angular momentum after correction of %10.10e: %10.10e\n",_momAng,_momAngf);
    }
#endif
}

void IF2D_ObstacleOperator::_parseArguments(ArgumentParser & parser)
{
    parser.set_strict_mode();
    const Real xpos = parser("-xpos").asDouble();
    parser.unset_strict_mode();
    const Real ypos = parser("-ypos").asDouble(0.5);
    angle = parser("-angle").asDouble(0.0);
    
    position[0] = xpos;
    position[1] = ypos;
}

void IF2D_ObstacleOperator::_writeComputedVelToFile(const int step_id, const Real t)
{
    const std::string fname = "computedVelocity_"+std::to_string(obstacleID)+".dat";
    std::ofstream savestream(fname, ios::out | ios::app);
    const std::string tab("\t");
    
    if(step_id==0)
        savestream << "step" << tab << "time" << tab << "CMx" << tab << "CMy" << tab << "angle" << tab << "vel_x" << tab << "vel_y" << tab << "angvel" << tab << "area"  << tab << "J" << std::endl;
    
    savestream << step_id << tab;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<float>::digits10 + 1);
    savestream << t << tab << position[0] << tab << position[1] << tab << angle << tab << transVel[0] << tab << transVel[1] << tab << angVel << tab << area << tab << J << std::endl;
    savestream.close();
}

void IF2D_ObstacleOperator::_writeDiagForcesToFile(const int step_id, const Real t)
{
    const std::string forcefilename = "diagnosticsForces_"+std::to_string(obstacleID)+".dat";
    std::ofstream savestream(forcefilename, ios::out | ios::app);
    const std::string tab("\t");
    
    if(step_id==0)
        savestream << "step" << tab << "time" << tab << "mass" << tab << "force_x" << tab << "force_y" << tab << "torque" << std::endl;
    
    savestream << step_id << tab;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<float>::digits10 + 1);
    savestream << t << tab << mass << tab << force[0] << tab << force[1] << tab << torque << std::endl;
    savestream.close();
}

void IF2D_ObstacleOperator::computeDiagnostics(const int stepID, const double time, const double* Uinf, const double lambda)
{
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();
    const double h = vInfo[0].h_gridpoint;
    
    double _area(0.0), _forcex(0.0), _forcey(0.0), _torque(0.0);
    #pragma omp parallel for schedule(static) reduction(+:_area,_forcex,_forcey,_torque)
    for(int i=0; i<vInfo.size(); i++)
    {
        std::map<int,ObstacleBlock*>::const_iterator pos = obstacleBlocks.find(i);
        if(pos == obstacleBlocks.end()) continue;
        
        BlockInfo info = vInfo[i];
        FluidBlock& b = *(FluidBlock*)info.ptrBlock;
        
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            const double Xs = pos->second->chi[iy][ix];
            if (Xs == 0) continue;
            
            double p[2];
            info.pos(p, ix, iy);
            p[0]-=position[0];
            p[1]-=position[1];
            
            const double object_UR[2] = {
                -angVel*p[1],
                 angVel*p[0]
            };
            
            const double object_UDEF[2] = {
                pos->second->udef[iy][ix][0],
                pos->second->udef[iy][ix][1],
            };
            
            const double U[2] = {
                b(ix,iy).u + Uinf[0] - (transVel[0]+object_UR[0]+object_UDEF[0]),
                b(ix,iy).v + Uinf[1] - (transVel[1]+object_UR[1]+object_UDEF[1])
            };
            
            _area += Xs;
            _forcex += U[0]*Xs;
            _forcey += U[1]*Xs;
            _torque += (p[0]*U[1]-p[1]*U[0])*Xs;
        }
    }
    
    const double dA = h*h;
    mass     = _area  *dA;
    force[0] = _forcex*dA*lambda;
    force[1] = _forcey*dA*lambda;
    torque   = _torque*dA*lambda;
    
    _writeDiagForcesToFile(stepID, time);
}

void IF2D_ObstacleOperator::computeVelocities(const double* Uinf)
{
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();
    double CM[2];
    this->getCenterOfMass(CM);
    
    double _A(0.0), _J(0.0), _momX(0.0), _momY(0.0), _momAng(0.0);
    #pragma omp parallel for schedule(static) reduction(+:_A,_J,_momX,_momY,_momAng)
    for(int i=0; i<vInfo.size(); i++)
    {
        std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);
        if(pos == obstacleBlocks.end()) continue;
            
        BlockInfo info = vInfo[i];
        FluidBlock& b = *(FluidBlock*)info.ptrBlock;
        
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
            const double Xs = pos->second->chi[iy][ix];
            if (Xs == 0) continue;
            double p[2];
            info.pos(p, ix, iy);
            p[0]-=CM[0];
            p[1]-=CM[1];
            
            _A      += Xs;
            _momX   += Xs * b(ix,iy).u;
            _momY   += Xs * b(ix,iy).v;
            _momAng += Xs * (p[0]*b(ix,iy).v - p[1]*b(ix,iy).u);
            _J      += Xs * (p[0]*p[0] + p[1]*p[1]);
        }
    }
    
    const double da = vInfo[0].h_gridpoint*vInfo[0].h_gridpoint;
    area        = _A * da;
    J           = _J * da;
    transVel[0] = _momX/_A + Uinf[0];
    transVel[1] = _momY/_A + Uinf[1];
    angVel      = _momAng/_J;
}

void IF2D_ObstacleOperator::computeForces(const int stepID, const double time, const double* Uinf, const double NU, const bool bDump)
{
    BlockInfo * ary = &vInfo.front();
    const int N = vInfo.size();
    Real vel_unit[2] = {0.0, 0.0};
    const Real velx_tot = transVel[0]-Uinf[0];
    const Real vely_tot = transVel[1]-Uinf[1];
    const Real vel_norm = std::sqrt(velx_tot*velx_tot + vely_tot*vely_tot);
    if (vel_norm>1e-9) {
        vel_unit[0] = (transVel[0]-Uinf[0])/vel_norm;
        vel_unit[1] = (transVel[1]-Uinf[1])/vel_norm;
    }
    
    const int stencil_start[3] = {-1,-1, 0};
    const int stencil_end[3]   = { 2, 2, 1};
    
    //surfData is processed serially, so points are ordered by block
    vector<int> usefulIDs; //which blocks are occupied by surface
    vector<int> firstInfo; //which entries in surfData correspond to each
    for(int i=0; i<surfData.Ndata; i++) {
        bool unique(true); //if Ive already seen that ID then skip
        for(int k=0; k<usefulIDs.size(); k++)
            if (surfData.Set[i]->blockID == usefulIDs[k])
            { unique = false; break; }
        
        if (unique) {
            usefulIDs.push_back(surfData.Set[i]->blockID);
            firstInfo.push_back(i);
        }
    }
    firstInfo.push_back(surfData.Ndata);
    
    double _totChi(0.0),_totFx(0.0),_totFy(0.0),_totFxP(0.0),_totFyP(0.0),_totFxV(0.0),_totFyV(0.0),_drag(0.0),_thrust(0.0),_Pout(0.0),_PoutBnd(0.0),_defPower(0.0),_defPowerBnd(0.0);
    #pragma omp parallel
    {
        Lab lab;
        lab.prepare(*grid, stencil_start, stencil_end, true);
        
        #pragma omp for schedule(static) reduction(+:_totChi,_totFx,_totFy,_totFxP,_totFyP,_totFxV,_totFyV,_drag,_thrust,_Pout,_PoutBnd,_defPower,_defPowerBnd)
        for (int j=0; j<usefulIDs.size(); j++) {
            const int k = usefulIDs[j];
            lab.load(ary[k], 0);
            
            BlockInfo info = vInfo[k];
            //FluidBlock& b = *(FluidBlock*)info.ptrBlock;
            
            const double _h2 = info.h_gridpoint*info.h_gridpoint;
            const double _1oH = NU / info.h_gridpoint; // 2 nu / 2 h
            
            for(int i=firstInfo[j]; i<firstInfo[j+1]; i++)
            {
                double p[2];
                const int ix = surfData.Set[i]->ix;
                const int iy = surfData.Set[i]->iy;
                const auto tempIt = obstacleBlocks.find(k);
                assert(tempIt != obstacleBlocks.end());
                info.pos(p, ix, iy);
                
                const double D11 =    _1oH*(lab(ix+1,iy).u - lab(ix-1,iy).u);
                const double D12 = .5*_1oH*(lab(ix,iy+1).u - lab(ix,iy-1).u
                                           +lab(ix+1,iy).v - lab(ix-1,iy).v);
                const double D22 =    _1oH*(lab(ix,iy+1).v - lab(ix,iy-1).v);
                const double normX = surfData.Set[i]->dchidx * _h2;
                const double normY = surfData.Set[i]->dchidy * _h2;
                const double fXV = D11 * normX + D12 * normY;
                const double fYV = D12 * normX + D22 * normY;
                const double fXP = -lab(ix,iy).p * normX;
                const double fYP = -lab(ix,iy).p * normY;
                const double fXT = fXV+fXP;
                const double fYT = fYV+fYP;
                
                surfData.P[i]  = lab(ix,iy).p;
                surfData.fX[i] = fXT;  surfData.fY[i] = fYT;
                surfData.fxP[i] = fXP; surfData.fyP[i] = fYP;
                surfData.fxV[i] = fXV; surfData.fyV[i] = fYV;
                surfData.pX[i] = p[0]; surfData.pY[i] = p[1];
                
                _totChi += surfData.Set[i]->delta * _h2;
                _totFxP += fXP; _totFyP += fYP;
                _totFxV += fXV; _totFyV += fYV;
                _totFx  += fXT; _totFy  += fYT;
                
                
                const double forcePar = fXT*vel_unit[0] + fYT*vel_unit[1];
                _thrust += .5*(forcePar + std::abs(forcePar));
                _drag   -= .5*(forcePar - std::abs(forcePar));
                
                surfData.vxDef[i] = tempIt->second->udef[iy][ix][0];
                surfData.vyDef[i] = tempIt->second->udef[iy][ix][1];
                surfData.vx[i] = lab(ix,iy).u + Uinf[0];
                surfData.vy[i] = lab(ix,iy).v + Uinf[1];
                
                const double powOut = fXT*surfData.vx[i] + fYT*surfData.vy[i];
                _Pout   += powOut;
                _PoutBnd+= min(0., powOut);
                
                const double powDef = fXT*surfData.vxDef[i] + fYT*surfData.vyDef[i];
                _defPower   += powDef;
                _defPowerBnd+= min(0., powDef);
            }
        }
    }

    totChi=_totChi;
    totFx=_totFx; totFy=_totFy;
    drag=_drag; thrust=_thrust;
    Pout=_Pout; PoutBnd=_PoutBnd;
    defPower=_defPower; defPowerBnd=_defPowerBnd;
    Pthrust    = thrust*vel_norm;
    Pdrag      =   drag*vel_norm;
    EffPDef    = Pthrust/(Pthrust-min(defPower,0.));
    EffPDefBnd = Pthrust/(Pthrust-    defPowerBnd);
    
    if (bDump)  surfData.print(obstacleID, stepID);
    if (bDump && pX not_eq nullptr && pX not_eq nullptr && Npts not_eq 0) {
    	surfData.sort(pX,pY,Npts);
    	surfData.printSorted(obstacleID, stepID);
    }
    
    {
        ofstream filedrag;
        filedrag.open(("forceValues_"+std::to_string(obstacleID)+".txt").c_str(), ios::app);
        filedrag<<time<<" "<<_totFxP<<" "<<_totFyP<<" "<<_totFxV<<" "<<_totFyV<<" "<<totFx<<" "<<totFy<<" "<<drag<<" "<<thrust<<" "<<totChi<<endl;
        filedrag.close();
    }
    {
        ofstream filedrag;
        filedrag.open(("powerValues_"+std::to_string(obstacleID)+".txt").c_str(), ios::app);
        filedrag<<time<<" "<<Pthrust<<" "<<Pdrag<<" "<<PoutBnd<<" "<<Pout<<" "<<defPowerBnd<<" "<<defPower<<" "<<EffPDefBnd<<" "<<EffPDef<<endl;
        filedrag.close();
    }
}

void IF2D_ObstacleOperator::update(const int step_id, const Real t, const Real dt)
{
    //Real velx_tot = transVel[0]-Uinf[0];
    //Real vely_tot = transVel[1]-Uinf[1];
    position[0] += dt*transVel[0];
    position[1] += dt*transVel[1];
    //ext_pos[0] += dt*velx_tot;
    //ext_pos[1] += dt*vely_tot;
    angle += dt*angVel;
    
    _writeComputedVelToFile(step_id, t);
}

void IF2D_ObstacleOperator::characteristic_function()
{
#pragma omp parallel
	{
		BlockInfo * ary = &vInfo.front();
		//CopyChiFromObstBlocks copychi();
#pragma omp for schedule(static)
		for(int i=0; i<vInfo.size(); i++)
		{
			std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);

			if(pos != obstacleBlocks.end()) {
				FluidBlock& b = *(FluidBlock*)ary[i].ptrBlock;
				const ObstacleBlock* const o = pos->second;
				for(int iy=0; iy<FluidBlock::sizeY; iy++)
					for(int ix=0; ix<FluidBlock::sizeX; ix++)
						b(ix,iy).chi = std::max(o->chi[iy][ix], b(ix,iy).chi);
			}
		}
	}
}

std::vector<int> IF2D_ObstacleOperator::intersectingBlockIDs(const int buffer)
{
	assert(buffer <= 2); // only works for 2: if different definition of deformation blocks, implement your own
	std::vector<int> retval;
	const int N = vInfo.size();

	for(int i=0; i<N; i++) {
		std::map<int,ObstacleBlock* >::const_iterator pos = obstacleBlocks.find(i);
		if(pos != obstacleBlocks.end())
			retval.push_back(i);
	}
	return retval;
}

void IF2D_ObstacleOperator::getTranslationVelocity(Real UT[2]) const
{
    UT[0]=transVel[0];
    UT[1]=transVel[1];
}

void IF2D_ObstacleOperator::setTranslationVelocity(Real UT[2])
{
    transVel[0]=UT[0];
    transVel[1]=UT[1];
}

void IF2D_ObstacleOperator::getAngularVelocity(Real & W) const
{
    W = angVel;
}

void IF2D_ObstacleOperator::setAngularVelocity(const Real W)
{
    angVel = W;
}

void IF2D_ObstacleOperator::getCenterOfMass(Real CM[2]) const
{
    CM[0]=position[0];
    CM[1]=position[1];
}

double IF2D_ObstacleOperator::getForceX() const
{
    return force[0];
}
double IF2D_ObstacleOperator::getForceY() const
{
    return force[1];
}

void IF2D_ObstacleOperator::save(const int step_id, const Real t, std::string filename)
{
    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<Real>::digits10 + 1);
    savestream.open(filename+".txt");
    savestream << t << std::endl;
    savestream << position[0] << "\t" << position[1] << std::endl;
    savestream << angle << std::endl;
    savestream << transVel[0] << "\t" << transVel[1] << std::endl;
    savestream << angVel << std::endl;
}

void IF2D_ObstacleOperator::restart(const Real t, std::string filename)
{
    std::ifstream restartstream;
    
    if(filename==std::string())
        restartstream.open("restart_IF2D_MovingBody.txt");
    else
        restartstream.open(filename+".txt");
    
    Real restart_time;
    restartstream >> restart_time;
    assert(std::abs(restart_time-t) < std::numeric_limits<Real>::epsilon());
    
    restartstream >> position[0] >> position[1];
    restartstream >> angle;
    restartstream >> transVel[0] >> transVel[1];
    restartstream >> angVel;
    restartstream.close();
    
    {
        std::cout << "RESTARTED BODY: " << std::endl;
        std::cout << "TIME: \t" << restart_time << std::endl;
        std::cout << "POS : \t" << position[0] << " " << position[1] << std::endl;
        std::cout << "ANGLE:\t" << angle << std::endl;
        std::cout << "TVEL: \t" << transVel[0] << " " << transVel[1] << std::endl;
        std::cout << "AVEL: \t" << angVel << std::endl;
    }
}

