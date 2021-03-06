cmake_minimum_required(VERSION 2.7)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")

add_library(emvisi2_lib
            emvisi2.h fwncc.h   growmat.h imstat.h gcgraph.hpp
            emvisi2.cpp fwncc.cpp     growmat.cpp        ncc_proba.cpp)
target_link_libraries(emvisi2_lib ${OpenCV_LIBS})

add_executable(emvisi2 emvisi2_test.cpp)
target_link_libraries(emvisi2 emvisi2_lib)

add_executable(emvisi2_video emvisi2_video.cpp)
target_link_libraries(emvisi2_video
		emvisi2_lib
		${Boost_SYSTEM_LIBRARY}
		${Boost_FILESYSTEM_LIBRARY}
		)

add_executable(learn learn.cpp)
target_link_libraries(learn emvisi2_lib)
