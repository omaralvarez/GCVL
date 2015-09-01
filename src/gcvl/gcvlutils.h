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

#pragma once

//! Macro that obtains unsigned integer division rounded up.
/*!
  \param a numerator.
  \param b denomitator.
  \return result.
*/
#define roundUpDiv(a,b) (a + b - 1) / b

//! Function that prints a character a certain number of times.
/*!
  \param x sentence that needs to be repeated.
  \param N times that it will be repeated.
  \param newline value that determines if the newline character should be placed at the end.
  \return repeated string.
*/
static void Print_N_Times(const std::string x, const int N, const bool newline = true)
{
    for (int i = 0 ; i < N ; i++)
    {
        std::cout << x;
    }

    if (newline)
        std::cout << std::endl;
}
