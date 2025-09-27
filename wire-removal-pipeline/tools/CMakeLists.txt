cmake_minimum_required(VERSION 3.18)
project(mask_exchange_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
# Ensure pkg-config points into your L4T sysroot (export these before cmake):
#   export PKG_CONFIG_SYSROOT_DIR=/l4t/targetfs
#   export PKG_CONFIG_LIBDIR=/l4t/targetfs/usr/lib/aarch64-linux-gnu/pkgconfig:/l4t/targetfs/usr/lib/pkgconfig:/l4t/targetfs/usr/share/pkgconfig
#   export PKG_CONFIG_PATH=${PKG_CONFIG_LIBDIR}
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0)

add_executable(mask_exchange_server mask_exchange_server.cpp)
target_include_directories(mask_exchange_server PRIVATE
    ${GST_INCLUDE_DIRS}
    ${CMAKE_SYSROOT}/usr/include
    ${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu
    ../../common)
target_link_directories(mask_exchange_server PRIVATE ${GST_LIBRARY_DIRS})
target_link_libraries(mask_exchange_server PRIVATE ${GST_LIBRARIES})
