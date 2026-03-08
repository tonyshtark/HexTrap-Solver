cmake_minimum_required(VERSION 3.15)
project(cpp_hex_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# 查找 Python 和 pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # 备选：通过 Python 获取 pybind11
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 CONFIG REQUIRED)
endif()

# 源文件
set(SOURCES hex_engine.cpp bindings.cpp)

# 构建 Python 扩展模块
pybind11_add_module(cpp_hex_engine ${SOURCES})

# 包含目录
target_include_directories(cpp_hex_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 安装到项目根目录（便于 Python 导入）
install(TARGETS cpp_hex_engine
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
