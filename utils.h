/*
 * Copyright (C) 2013 by
 * 
 * Cheng Zhang
 * chengz@kth.se
 * and
 * Xavi Gratal
 * javiergm@kth.se
 * Computer Vision and Active Perception Lab
 * KTH Royal Institue of Techonology
 *
 * onlineHDP_C++ is a free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *
 * onlineHDP_C++ is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with onlineHDP_C++; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
 */

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include "onlinehdp.h"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace HDP {

// parse command line arguments
int parse_args(int argc, char ** argv, model & pmodel);

int getdir(std::string dir, std::vector<std::string> &files );
Eigen::MatrixXd log_normalize(const Eigen::MatrixXd & v);

double log_sum(double a, double b);

int argmax(const Eigen::VectorXd & x);

template<typename type>
std::string convert(type number)
{
    std::stringstream ss;
    ss<<number;
    return ss.str();
}

double digamma(double x);
double loggamma(double x);

}

#endif
