cmake_minimum_required(VERSION 2.7)

find_package(OpenCV REQUIRED)

add_library(emvisi2_lib
            emvisi2.h fwncc.h   growmat.h imstat.h
            emvisi2.cpp fwncc.cpp     growmat.cpp        ncc_proba.cpp)
target_link_libraries(emvisi2_lib ${OpenCV_LIBS})

add_executable(emvisi2 emvisi2_test.cpp)
target_link_libraries(emvisi2 emvisi2_lib)

add_executable(learn learn.cpp)
target_link_libraries(learn emvisi2_lib)
