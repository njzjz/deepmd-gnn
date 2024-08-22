file(GLOB OP_SRC mace.cc)

add_library(deepmd_mace MODULE ${OP_SRC})
# link: libdeepmd libtorch
target_link_libraries(deepmd_mace PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(
    deepmd_mace
  PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=${OP_CXX_ABI_PT}>")
if(APPLE)
  set_target_properties(deepmd_mace PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(deepmd_mace PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

if(BUILD_PY_IF)
  install(TARGETS deepmd_mace DESTINATION deepmd_mace/lib/)
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION deepmd_mace/lib)
else(BUILD_PY_IF)
  install(TARGETS deepmd_mace DESTINATION lib/)
endif(BUILD_PY_IF)
