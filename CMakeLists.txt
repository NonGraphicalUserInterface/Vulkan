cmake_minimum_required(VERSION 3.16)
project(VulkanTutorial)

set(CMAKE_CXX_STANDARD 17)

set ( GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only")

set ( GLFW_INSTALL OFF CACHE BOOL "GLFW lib only")

add_subdirectory( glfw )
add_executable(VulkanTutorial main.cpp util/util.cpp util/camera.cpp)

find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)

find_package(glm REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${GLM_LIBRARY_DIRS})

target_link_libraries(${PROJECT_NAME} glfw)

include_directories("E:/C++/Projects/Libraries/include")
include_directories("E:/C++/Projects/VulkanTutorial")

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/vert.spv ${CMAKE_BINARY_DIR}/frag.spv
    COMMAND ${CMAKE_COMMAND} -E echo "Running compile.bat"
    COMMAND compile.bat
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running compile.bat"
)
