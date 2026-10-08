# SPDX-License-Identifier: GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Anirban Phukan
# reaxmetal_embed_text(<in-file> <out-inc>): writes a C++ raw string literal with the file's text, so that a shader source can be
# compiled into the library and handed to the Metal runtime compiler. Re-runs CMake when the input changes.
function(reaxmetal_embed_text in out)
  file(READ "${in}" _text)
  if(_text MATCHES "\\)RMSRC\"")
    message(FATAL_ERROR "${in} contains the raw-string terminator )RMSRC\"")
  endif()
  file(WRITE "${out}.tmp" "R\"RMSRC(${_text})RMSRC\"\n")
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${out}.tmp" "${out}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${in}")
endfunction()
