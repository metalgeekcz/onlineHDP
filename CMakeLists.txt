cmake_minimum_required(VERSION 2.8)    
cmake_policy(VERSION 2.8.0)

project(onlineHDP)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++0x)
include_directories(/usr/local/include/eigen3)

set(cc_sources onlinehdp.cpp corpus.cpp run_online_hdp.cpp utils.cpp)
add_executable(onlinehdp ${cc_sources})
