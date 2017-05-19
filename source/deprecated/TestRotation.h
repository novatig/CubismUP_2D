//
//  TestRotation.h
//  CubismUP_2D
//
//	Impose rotation and check that the direction and angular velocities are correct
//
//  Created by Christian Conti on 3/16/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef __CubismUP_2D__TestRotation__
#define __CubismUP_2D__TestRotation__

#include <stdio.h>
#include "Test.h"
#include "Shape.h"

class TestRotation : public Test
{
private:
	int bpd;
	Real omegaBody;
	Shape * shape;
	const int testCase; // 0: forced, 1: from flow
	double dt;
	
	string path2file;
	SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;
	
	FluidGrid * grid;
	
	void _ic();
	
public:
	TestRotation(const int argc, const char ** argv, const int testCase, const int bpd, const double dt);
	~TestRotation();
	
	void run();
	void check();
};

#endif /* defined(__CubismUP_2D__TestRotation__) */
