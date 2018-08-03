//
//  IF2D_CarlingFishOperator.h
//  IF2D_ROCKS
//
//  Created by Wim van Rees on 08/10/14.
//
//

#ifndef __IF2D_ROCKS__IF2D_CarlingFishOperator__
#define __IF2D_ROCKS__IF2D_CarlingFishOperator__

#include "IF2D_FishOperator.h"

class IF2D_CarlingFishOperator: public IF2D_FishOperator
{
protected:

    void _parseArguments(ArgumentParser& parser)
    {
        printf("created IF2D_CarlingFish: xpos=%3.3f ypos=%3.3f angle=%3.3f L=%3.3f T=%3.3f phi=%3.3f\n",position[0],position[1],angle,length,Tperiod,phaseShift);
        parser.unset_strict_mode();
    }
    
    void _initializeFishData();
    
public:
    
    IF2D_CarlingFishOperator(FluidGrid * grid, ArgumentParser& parser):IF2D_FishOperator(grid,parser)
    {
        _parseArguments(parser);
        _initializeFishData();        
    }
    void save(const int step_id, const double t, std::string filename = std::string()) override;
    void restart(const double t, std::string filename = std::string()) override;

};


#endif /* defined(__IF2D_ROCKS__IF2D_CarlingFishOperator__) */
