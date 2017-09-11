cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(supervoxel_densecrf)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_ply_view my_ply_pcd_visualization.cpp)
target_link_libraries (pcd_ply_view ${PCL_LIBRARIES})