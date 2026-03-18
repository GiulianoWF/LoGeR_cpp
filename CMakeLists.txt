cmake_minimum_required(VERSION 3.18)
project(loger_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json so clangd/IDE can resolve include paths
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# --- Dependencies ---
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

# --- Library ---
add_library(loger_lib STATIC
    src/io/weight_loader.cpp
    src/io/image_loader.cpp
    src/io/output_writer.cpp
    src/ops/rope2d.cpp
    src/ops/attention.cpp
    src/model/encoder.cpp
    src/model/block_rope.cpp
    src/model/ttt.cpp
    src/model/task_heads.cpp
    src/model/pi3.cpp
    src/utils/geometry.cpp
    src/utils/windowing.cpp
)

target_include_directories(loger_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(loger_lib PUBLIC
    "${TORCH_LIBRARIES}"
    ${OpenCV_LIBS}
)

target_compile_options(loger_lib PUBLIC ${TORCH_CXX_FLAGS})

# --- Main binary ---
add_executable(loger_infer app/main.cpp)
target_link_libraries(loger_infer PRIVATE loger_lib)

# --- Tests ---
enable_testing()

foreach(test_name rope2d encoder ttt e2e)
    add_executable(test_${test_name} tests/test_${test_name}.cpp)
    target_link_libraries(test_${test_name} PRIVATE loger_lib)
    add_test(NAME ${test_name} COMMAND test_${test_name})
endforeach()

# Copy torch DLLs on Windows (no-op on Linux)
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET loger_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS} $<TARGET_FILE_DIR:loger_infer>)
endif()
