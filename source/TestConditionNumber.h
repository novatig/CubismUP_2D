//
//  TestConditionNumber.h
//  CubismUP_2D
//
//  Created by Christian Conti on 11/8/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef __CubismUP_2D__TestConditionNumber__
#define __CubismUP_2D__TestConditionNumber__

#include "Test.h"
#include "GenericCoordinator.h"
#include "Shape.h"

class TestConditionNumber : public Test
{
protected:
	ArgumentParser parser;
	
	double rhoS;
	const int bpd;
	
	string path2file;
	SerializerIO_ImageVTK<FluidGrid, FluidVTKStreamer> dumper;
	
	FluidGrid * grid;
	
	// body
	Shape * shape;
	
	void _mean(Real c, Real e, Real w, Real n, Real s, Real& avgE, Real& avgW, Real& avgN, Real& avgS);
	void _ic();
	
public:
	TestConditionNumber(const int argc, const char ** argv, const int bpd);
	~TestConditionNumber();
	
	void run();
	void check();
};


#endif /* defined(__CubismUP_2D__TestConditionNumber__) */
