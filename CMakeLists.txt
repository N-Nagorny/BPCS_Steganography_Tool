cmake_minimum_required(VERSION 3.5)

project(BPCS LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Core Gui Widgets REQUIRED)
find_package(OpenCV REQUIRED)

set(SOURCES
    include/mainwindow.h
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.ui
)

set(HEADERS
    include/
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${HEADERS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    ${OpenCV_LIBS}
)
