# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/Desktop/findCircles/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/Desktop/findCircles/cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/ZED_with_OpenCV.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ZED_with_OpenCV.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ZED_with_OpenCV.dir/flags.make

CMakeFiles/ZED_with_OpenCV.dir/src/main.o: CMakeFiles/ZED_with_OpenCV.dir/flags.make
CMakeFiles/ZED_with_OpenCV.dir/src/main.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/Desktop/findCircles/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ZED_with_OpenCV.dir/src/main.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ZED_with_OpenCV.dir/src/main.o -c /home/nvidia/Desktop/findCircles/cpp/src/main.cpp

CMakeFiles/ZED_with_OpenCV.dir/src/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ZED_with_OpenCV.dir/src/main.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nvidia/Desktop/findCircles/cpp/src/main.cpp > CMakeFiles/ZED_with_OpenCV.dir/src/main.i

CMakeFiles/ZED_with_OpenCV.dir/src/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ZED_with_OpenCV.dir/src/main.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nvidia/Desktop/findCircles/cpp/src/main.cpp -o CMakeFiles/ZED_with_OpenCV.dir/src/main.s

CMakeFiles/ZED_with_OpenCV.dir/src/main.o.requires:

.PHONY : CMakeFiles/ZED_with_OpenCV.dir/src/main.o.requires

CMakeFiles/ZED_with_OpenCV.dir/src/main.o.provides: CMakeFiles/ZED_with_OpenCV.dir/src/main.o.requires
	$(MAKE) -f CMakeFiles/ZED_with_OpenCV.dir/build.make CMakeFiles/ZED_with_OpenCV.dir/src/main.o.provides.build
.PHONY : CMakeFiles/ZED_with_OpenCV.dir/src/main.o.provides

CMakeFiles/ZED_with_OpenCV.dir/src/main.o.provides.build: CMakeFiles/ZED_with_OpenCV.dir/src/main.o


# Object files for target ZED_with_OpenCV
ZED_with_OpenCV_OBJECTS = \
"CMakeFiles/ZED_with_OpenCV.dir/src/main.o"

# External object files for target ZED_with_OpenCV
ZED_with_OpenCV_EXTERNAL_OBJECTS =

ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/src/main.o
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/build.make
ZED_with_OpenCV: /usr/local/zed/lib/libsl_input.so
ZED_with_OpenCV: /usr/local/zed/lib/libsl_core.so
ZED_with_OpenCV: /usr/local/zed/lib/libsl_zed.so
ZED_with_OpenCV: /usr/local/lib/libopencv_video.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_photo.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_objdetect.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_gapi.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_dnn.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_stitching.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_ml.so.4.0.0
ZED_with_OpenCV: /usr/lib/aarch64-linux-gnu/libcuda.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libcudart.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppial.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppisu.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppicc.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppicom.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppidei.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppif.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppig.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppim.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppist.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnppitc.so
ZED_with_OpenCV: /usr/local/cuda-9.0/lib64/libnpps.so
ZED_with_OpenCV: /usr/local/lib/libopencv_calib3d.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_features2d.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_highgui.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_videoio.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_imgcodecs.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_imgproc.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_flann.so.4.0.0
ZED_with_OpenCV: /usr/local/lib/libopencv_core.so.4.0.0
ZED_with_OpenCV: CMakeFiles/ZED_with_OpenCV.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/Desktop/findCircles/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ZED_with_OpenCV"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ZED_with_OpenCV.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ZED_with_OpenCV.dir/build: ZED_with_OpenCV

.PHONY : CMakeFiles/ZED_with_OpenCV.dir/build

CMakeFiles/ZED_with_OpenCV.dir/requires: CMakeFiles/ZED_with_OpenCV.dir/src/main.o.requires

.PHONY : CMakeFiles/ZED_with_OpenCV.dir/requires

CMakeFiles/ZED_with_OpenCV.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ZED_with_OpenCV.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ZED_with_OpenCV.dir/clean

CMakeFiles/ZED_with_OpenCV.dir/depend:
	cd /home/nvidia/Desktop/findCircles/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/Desktop/findCircles/cpp /home/nvidia/Desktop/findCircles/cpp /home/nvidia/Desktop/findCircles/cpp/build /home/nvidia/Desktop/findCircles/cpp/build /home/nvidia/Desktop/findCircles/cpp/build/CMakeFiles/ZED_with_OpenCV.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ZED_with_OpenCV.dir/depend

