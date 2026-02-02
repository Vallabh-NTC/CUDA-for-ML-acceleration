cmake_minimum_required(VERSION 3.18)
project(vpi_dense_of_nvbuffer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vpi_dense_of_nvbuffer
    src/vpi_dense_of_nvbuffer.cpp
)

target_include_directories(vpi_dense_of_nvbuffer PRIVATE
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    ${CMAKE_SYSROOT}/usr/include/gstreamer-1.0
    ${CMAKE_SYSROOT}/usr/include/glib-2.0
    ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/glib-2.0/include
    ${CMAKE_SYSROOT}/usr/src/jetson_multimedia_api/include
    ${CMAKE_SYSROOT}/opt/nvidia/vpi2/include
)

set(SYS_LIB_DIR   ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu)
set(TEGRA_LIB_DIR ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/tegra)
set(VPI_LIB_DIR   ${CMAKE_SYSROOT}/opt/nvidia/vpi2/lib/aarch64-linux-gnu)

find_library(GST_LIB      gstreamer-1.0 HINTS ${SYS_LIB_DIR})
find_library(GST_BASE_LIB gstbase-1.0   HINTS ${SYS_LIB_DIR})
find_library(GST_APP_LIB  gstapp-1.0    HINTS ${SYS_LIB_DIR})
find_library(GLIB_LIB     glib-2.0      HINTS ${SYS_LIB_DIR})
find_library(GOBJECT_LIB  gobject-2.0   HINTS ${SYS_LIB_DIR})

find_library(NVBUFSURFACE_LIB nvbufsurface HINTS ${TEGRA_LIB_DIR} ${SYS_LIB_DIR})
find_library(VPI_LIB nvvpi HINTS ${VPI_LIB_DIR} ${SYS_LIB_DIR} ${TEGRA_LIB_DIR})

if(NOT VPI_LIB)
  message(FATAL_ERROR "libnvvpi.so not found in sysroot under ${VPI_LIB_DIR}")
endif()

target_link_libraries(vpi_dense_of_nvbuffer PRIVATE
    ${GST_LIB} ${GST_BASE_LIB} ${GST_APP_LIB}
    ${GLIB_LIB} ${GOBJECT_LIB}
    ${NVBUFSURFACE_LIB}
    ${VPI_LIB}
    pthread dl m
)

target_link_options(vpi_dense_of_nvbuffer PRIVATE
    -Wl,--no-as-needed
    -L${SYS_LIB_DIR}
    -L${TEGRA_LIB_DIR}
    -L${VPI_LIB_DIR}
    -Wl,-rpath,/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/opt/nvidia/vpi2/lib/aarch64-linux-gnu

    # Cross-link robustness:
    -Wl,--allow-shlib-undefined
    -Wl,--unresolved-symbols=ignore-in-shared-libs
)
