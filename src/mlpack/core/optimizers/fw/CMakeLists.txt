set(SOURCES
  atoms.hpp
  frank_wolfe.hpp
  frank_wolfe_impl.hpp
  constr_lpball.hpp
  constr_structure_group.hpp
  constr_matrix_lp.hpp
  update_classic.hpp
  update_span.hpp
  update_span_impl.hpp
  update_full_correction.hpp
  update_linesearch.hpp
  func_sq.hpp
  test_func_fw.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
