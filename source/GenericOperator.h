//
//  GenericOperator.h
//  CubismUP_2D
//
//  Created by Christian Conti on 3/6/15.
//  Copyright (c) 2015 ETHZ. All rights reserved.
//

#pragma once

class GenericOperator
{
public:
	virtual void operator()(const BlockInfo& info, FluidBlock& block) const = 0;
};

class GenericLabOperator
{
public:
  StencilInfo stencil;
	int stencil_start[3];
	int stencil_end[3];

	// cannot put the templated operator here!
};
