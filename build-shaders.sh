#!/bin/bash

glslc ./shaders/shader.glsl.frag -o ./shaders/shader.spv.frag
./sdl3-shadercross/bin/shadercross ./shaders/shader.spv.frag -o ./shaders/shader.msl.frag

glslc ./shaders/shader.glsl.vert -o ./shaders/shader.spv.vert
./sdl3-shadercross/bin/shadercross ./shaders/shader.spv.vert -o ./shaders/shader.msl.vert
