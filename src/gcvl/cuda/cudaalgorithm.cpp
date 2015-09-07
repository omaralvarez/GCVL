/*
 *
 * GPGPU Computer Vision Library (GCVL)
 *
 * Copyright (c) Luis Omar Alvarez Mures 2015 <omar.alvarez@udc.es>
 * Copyright (c) Emilio Padron Gonzalez 2015 <emilioj@gmail.com>
 *
 * All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this library.
 *
 */

#include "cudaalgorithm.h"

#include <boost/timer/timer.hpp>
#include <iostream>

using namespace gcvl::cuda;

//! Performs all the necessary steps to execute the algorithm.
/*!
	\return Computation time.
  \sa prepare(), setArgs(), launch() and postpare()
*/
double Algorithm::compute() {

    std::cout << " **** Starting! ****" << std::endl;

    prepare();
    setArgs();
	boost::timer::cpu_timer timer;
    launch();
    postpare();

    auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().user + timer.elapsed().system);
    auto microseconds = boost::chrono::duration_cast<boost::chrono::microseconds>(nanoseconds);

    std::cout << " **** Finished in " << microseconds.count()/1000000. << " s.! ****" << std::endl;

	return microseconds.count()/1000000.;

}
