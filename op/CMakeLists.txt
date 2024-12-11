file(GLOB OP_SRC edge_index.cc)

add_library(deepmd_gnn MODULE ${OP_SRC})
# link: libdeepmd libtorch
target_include_directories(deepmd_gnn PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(
  deepmd_gnn
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>")
if(APPLE)
  set_target_properties(deepmd_gnn PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(deepmd_gnn PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

if(BUILD_PY_IF)
  install(TARGETS deepmd_gnn DESTINATION deepmd_gnn/lib/)
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
          DESTINATION deepmd_gnn/lib)
else(BUILD_PY_IF)
  install(TARGETS deepmd_gnn DESTINATION lib/)
endif(BUILD_PY_IF)
