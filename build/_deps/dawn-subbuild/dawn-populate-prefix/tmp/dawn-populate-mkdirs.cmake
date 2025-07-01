# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-src")
  file(MAKE_DIRECTORY "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-src")
endif()
file(MAKE_DIRECTORY
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-build"
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix"
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/tmp"
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/src/dawn-populate-stamp"
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/src"
  "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/src/dawn-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/src/dawn-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "C:/Users/alder/WebGPU-Voxel-Engine/build/_deps/dawn-subbuild/dawn-populate-prefix/src/dawn-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
