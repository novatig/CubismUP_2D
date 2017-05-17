//
//  Test.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#ifndef CubismUP_2D_Test_h
#define CubismUP_2D_Test_h

#include <stdio.h>
#include "Definitions.h"

class Test
{
protected:
	ArgumentParser parser;
	
public:
	Test(const int argc, const char ** argv) : parser(argc,argv) {}
	virtual ~Test() {}
	
	virtual void run() = 0;
	virtual void check() = 0;
};

#endif
