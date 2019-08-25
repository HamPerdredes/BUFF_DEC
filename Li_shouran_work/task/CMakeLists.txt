# CMake 最低版本号要求
cmake_minimum_required (VERSION 2.8)
# 项目信息
project (buff)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV 3.4.5 REQUIRED)#4.0.0 REQUIRED)
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 指定生成目标
add_executable(buff ${DIR_SRCS})
target_link_libraries(buff ${OpenCV_LIBS})
