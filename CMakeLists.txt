cmake_minimum_required(VERSION 2.8)
project(BUFF)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
find_package(OpenCV 3.4.5 REQUIRED)
find_package(Threads REQUIRED)

include_directories(
${OpenCV_INCLUDE_DIRS}
)

include_directories(./include)

add_executable(Test "main.cpp")

AUX_SOURCE_DIRECTORY(src BUFF_DIR)

add_library(DETECTOR ${BUFF_DIR})

target_link_libraries(Test Threads::Threads DETECTOR ${OpenCV_LIBS} pthread)

