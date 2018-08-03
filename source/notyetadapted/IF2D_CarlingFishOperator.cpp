//
//  IF2D_CarlingFishOperator.cpp
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 08/10/14.
//
//

#include "IF2D_ObstacleLibrary.h"
#include "IF2D_CarlingFishOperator.h"
#include <array>
#include <cmath>
#include <time.h>

namespace FishObstacle
{    
    struct CarlingFishData : AmplitudeDefinedFishData
    {
    protected:

        const double sb, st, wt, wh;
        
        double midlineLateralPos(const double s, const Real t, const Real L,
                                 const Real T, const Real phaseShift) override
        {
            const double arg = 2.0*M_PI*(s/L - t/T + phaseShift);
            return 4./33. *  (s + 0.03125*L)*std::sin(arg);

        }
        
        double midlineLateralVel(const double s, const Real t, const Real L,
                                 const Real T, const Real phaseShift) override
        {
            const double arg = 2.0*M_PI*(s/L - t/T + phaseShift);
            return - 4./33. * (s + 0.03125*L) * (2.0*M_PI/T) * std::cos(arg);
        }
        
        void midlineLearnUpdate(const Real t, const Real T) override
        {
        }
        
        double _width(const double s, const Real L) override
        {
            if(s<0 or s>L)
                return 0;
            
            return (s<sb ? std::sqrt(2.0*wh*s-s*s) :
                    (s<st ? wh-(wh-wt)*std::pow((s-sb)/(st-sb),1) : // careful: pow(.,1) is 2D, pow(.,2) is 3D (Stefan paper)
                     (wt * (L-s)/(L-st))));
            
        }
        
    public:
        
        CarlingFishData(const int Nm, const Real length, const Real Tperiod, const Real phaseShift, const bool cosine=true): AmplitudeDefinedFishData(Nm,length,Tperiod,phaseShift,cosine),sb(0.04*length),st(0.95*length),wt(0.01*length),wh(0.04*length)
        {
            _computeWidth();
        }
        
        CarlingFishData(const int Nm, const Real length, const Real Tperiod, const Real phaseShift, const std::pair<int,Real> extension_info, const bool cosine=true): AmplitudeDefinedFishData(Nm,length,Tperiod,phaseShift,extension_info,cosine),sb(0.04*length),st(0.95*length),wt(0.01*length),wh(0.04*length)
        {
            _computeWidth();
        }
    };
}

void IF2D_CarlingFishOperator::_initializeFishData()
{
    const Real target_ds = 0.5*vInfo[0].h_gridpoint;
    const Real target_Nm = length/target_ds;
    
    // multiple of 100
    const int Nm = 100*(int)std::ceil(target_Nm/100) + 1;
    
    Nsegments = 20;
    assert((Nm-1)%Nsegments==0);
    
    // deal with extension
    const Real dx_extension = 0.25*vInfo[0].h_gridpoint;
    const int Nextension = 12;// up to 3dx on each side (to get proper interpolation up to 2dx)
    
    myFish = new FishObstacle::CarlingFishData(Nm, length, Tperiod, phaseShift, std::make_pair(Nextension,dx_extension));
}

void IF2D_CarlingFishOperator::save(const int step_id, const double t, string filename)
{

    std::ofstream savestream;
    savestream.setf(std::ios::scientific);
    savestream.precision(std::numeric_limits<double>::digits10 + 1);
    savestream.open(filename+".txt");
    savestream << t << std::endl;
    savestream << position[0] << "\t" << position[1] << std::endl;
    savestream << angle << std::endl;
    savestream << transVel[0] << "\t" << transVel[1] << std::endl;
    savestream << angVel << std::endl;
    savestream << theta_internal << "\t" << angvel_internal << std::endl;
    savestream.close();    
}

void IF2D_CarlingFishOperator::restart(const double t, string filename)
{
    std::ifstream restartstream;
    double read_time;
    restartstream.open(filename+".txt");
    restartstream >> read_time;
    //assert(std::abs(read_time-t) < std::numeric_limits<Real>::epsilon());
    
    restartstream >> position[0] >> position[1];
    restartstream >> angle;
    restartstream >> transVel[0] >> transVel[1];
    restartstream >> angVel;
    restartstream >> theta_internal >> angvel_internal;
    restartstream.close();
    
    {
        std::cout << "CARLING FISH: " << std::endl;
        std::cout << "TIME: \t" << t << std::endl;
        std::cout << "POS: \t" << position[0] << " " << position[1] << std::endl;
        std::cout << "ANGLE: \t" << angle << std::endl;
        std::cout << "TVEL: \t" << transVel[0] << " " << transVel[1] << std::endl;
        std::cout << "AVEL: \t" << angVel << std::endl;
        std::cout << "INTERN: \t" << theta_internal << " " << angvel_internal << std::endl;
    }
}
