cmake_minimum_required(VERSION 3.10)

project(nn_from_scratch VERSION 0.1.0 LANGUAGES CXX)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set debug flags for MSVC and others
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /DDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/DEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen
find_package (Eigen3 REQUIRED NO_MODULE)

# Automatically scan all source and header files in src/ and subdirectories
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.h" "src/*.hpp")

add_executable(nn_from_scratch ${SOURCES} ${HEADERS})

target_include_directories(nn_from_scratch PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(nn_from_scratch PRIVATE Eigen3::Eigen)

# Install target (optional)
install(TARGETS nn_from_scratch DESTINATION bin)