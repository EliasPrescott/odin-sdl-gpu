package main

import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/rand"
import "core:math/noise"
import "core:math/linalg"

import sdl "vendor:sdl3"
import "vendor:sdl3/image"

Vec3 :: [3]f32
Vec2 :: [2]f32

LOOK_SENSITIVITY :: 0.3

Camera :: struct {
    pos: Vec3,
    target: Vec3,
}

PlayerLookInfo :: struct {
    yaw: f32,
    pitch: f32,
}

AppState :: struct {
    camera: Camera,
    key_down: #sparse [sdl.Scancode]bool,
    mouse_move: Vec2,
    player_look_info: PlayerLookInfo,
}

handle_err :: proc(success: bool) {
    if !success {
        err := sdl.GetError()
        fmt.println(err)
        panic("quitting due to error")
    }
}

frag_shader_code := #load("./shaders/shader.msl.frag")
vert_shader_code := #load("./shaders/shader.msl.vert")

random_cell :: proc(base: Vec3) -> GridCell {
    return GridCell{
        p = {
            Vec3{0, 0, 0} + base,
            Vec3{1, 0, 0} + base,
            Vec3{1, 0, 1} + base,
            Vec3{0, 0, 1} + base,
            Vec3{0, 1, 0} + base,
            Vec3{1, 1, 0} + base,
            Vec3{1, 1, 1} + base,
            Vec3{0, 1, 1} + base,
        },
        val = {
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
            rand.float32_range(0, 1),
        },
    }
}

generate_grid :: proc(grid_dimensions: [3]int, threshold: f32) -> []Triangle {
    triangles: [dynamic]Triangle

    noise_cube := make(map[[3]int]f32, grid_dimensions.x * grid_dimensions.y * grid_dimensions.z)
    for x in 0..=grid_dimensions.x {
        for y in 0..=grid_dimensions.y {
            for z in 0..=grid_dimensions.z {
                coord := [3]int {x, y, z}
                noise := noise.noise_3d_improve_xz(10, {f64(x), f64(y), f64(z)})
                value := f32(y) + noise
                noise_cube[coord] = value
            }
        }
    }

    for x in 0..<grid_dimensions.x {
        for y in 0..<grid_dimensions.y {
            for z in 0..<grid_dimensions.z {
                base := Vec3{f32(x), f32(y), f32(z)}
                value := noise_cube[{x,y,z}]
                // fmt.printfln("noise(%v): %v", [3]int{x,y,x}, value)
                cell := GridCell {
                    p = {
                        base,
                        Vec3{1, 0, 0} + base,
                        Vec3{1, 0, 1} + base,
                        Vec3{0, 0, 1} + base,
                        Vec3{0, 1, 0} + base,
                        Vec3{1, 1, 0} + base,
                        Vec3{1, 1, 1} + base,
                        Vec3{0, 1, 1} + base,
                    },
                    val = {
                        noise_cube[{x,y,z}],
                        noise_cube[{x+1,y,z}],
                        noise_cube[{x+1,y,z+1}],
                        noise_cube[{x,y,z+1}],
                        noise_cube[{x,y+1,z}],
                        noise_cube[{x+1,y+1,z}],
                        noise_cube[{x+1,y+1,z+1}],
                        noise_cube[{x,y+1,z+1}],
                    },
                }
                local_triangles := [5]Triangle{}
                triangle_count := Polygonise(cell, threshold, local_triangles[:])
                for i in 0..<triangle_count {
                    append_elem(&triangles, local_triangles[i])
                }
            }
        }
    }

    return triangles[:]
}

make_test_grid :: proc() -> []Triangle {
    triangles := [dynamic]Triangle {}

    {
        test_cell := GridCell {
            p = {
                {0,0,0},
                {1,0,0},
                {1,0,1},
                {0,0,1},
                {0,1,0},
                {1,1,0},
                {1,1,1},
                {0,1,1},
            },
            val = {
                1, 1, 0, 0,
                1, 0, 0, 0,
            },
        }
        local_triangles := [5]Triangle{}
        triangle_count := Polygonise(test_cell, 0.5, local_triangles[:])
        for i in 0..<triangle_count {
            append_elem(&triangles, local_triangles[i])
        }
    }

    return triangles[:]
}

update_camera :: proc(state: ^AppState, delta: f32) {
    move_input: Vec3
    if state.key_down[.W] do move_input.y = 1
    else if state.key_down[.S] do move_input.y = -1
    if state.key_down[.A] do move_input.x = -1
    else if state.key_down[.D] do move_input.x = 1
    if state.key_down[.Q] do move_input.z = -1
    else if state.key_down[.E] do move_input.z = 1

    move_input = linalg.normalize0(move_input)

    look_input := state.mouse_move * LOOK_SENSITIVITY
    state.player_look_info.yaw = math.wrap(state.player_look_info.yaw - look_input.x, 360)
    state.player_look_info.pitch = math.clamp(state.player_look_info.pitch - look_input.y, -89, 89)

    look_matrix := linalg.matrix3_from_yaw_pitch_roll_f32(
        linalg.to_radians(state.player_look_info.yaw),
        linalg.to_radians(state.player_look_info.pitch),
        0
    )

    forward := look_matrix * Vec3 { 0, 0, -1 }
    right := look_matrix * Vec3 { 1, 0, 0 }
    up := look_matrix * Vec3 {0,1,0}
    move_dir := forward * move_input.y + right * move_input.x + up * move_input.z

    motion := move_dir * delta * MOVE_SPEED

    MOVE_SPEED :: 5
    state.camera.pos += motion
    state.camera.target = state.camera.pos + forward
}

calculate_triangle_normal :: proc(tri: Triangle) -> Vec3 {
    a := tri.p.y - tri.p.x
    b := tri.p.z - tri.p.x
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    }
}

main :: proc() {
    triangles := generate_grid({20,5,20}, 1)
    // triangles := make_test_grid()

    sdl.SetLogPriorities(.VERBOSE)
    handle_err(sdl.Init({.VIDEO}))
    window := sdl.CreateWindow("marching-cubes (ignore)", 1000, 800, {})

    // The shader formats here are the ones the program claims to have available.
    gpu := sdl.CreateGPUDevice({.MSL}, true, nil)

    // The formats here are the ones supported by the GPU.
    // If the GPU support doesn't overlap with what I claim to support,
    // then the previous CreateGPUDevice() call should fail.
    // To run this cross-platform, I'll need to read the results from GetGPUShaderFormats()
    // and use that to load the appropriate shaders.
    fmt.println(sdl.GetGPUShaderFormats(gpu))

    handle_err(sdl.ClaimWindowForGPUDevice(gpu, window))

    defer {
        sdl.DestroyGPUDevice(gpu)
        sdl.DestroyWindow(window)
        sdl.Quit()
    }

    handle_err(sdl.SetWindowRelativeMouseMode(window, true))

    state := AppState {
        camera = Camera {
            pos = {0, 0, 5},
            target = {0, 0, 0},
        },
    }

    cobblestone_texture := image.Load("./cobblestone_1.png")

    texture := sdl.CreateGPUTexture(gpu, {
        format = .R8G8B8A8_UNORM,
        usage = {.SAMPLER},
        width = u32(cobblestone_texture.w),
        height = u32(cobblestone_texture.h),
        layer_count_or_depth = 1,
        num_levels = 1,
    }) 


    win_size: [2]i32
    sdl.GetWindowSize(window, &win_size[0], &win_size[1])
    depth_texture := sdl.CreateGPUTexture(gpu, {
        format = .D16_UNORM,
        usage = {.DEPTH_STENCIL_TARGET},
        width = u32(win_size.x),
        height = u32(win_size.y),
        layer_count_or_depth = 1,
        num_levels = 1,
    }) 

    frag_shader := sdl.CreateGPUShader(gpu, {
        code_size = len(frag_shader_code),
        code = raw_data(frag_shader_code),
        // the cross-compilation process turns main into main0
        entrypoint = "main0",
        format = {.MSL},
        stage = .FRAGMENT,
        num_samplers = 1,
        num_uniform_buffers = 1,
    })

    vert_shader := sdl.CreateGPUShader(gpu, {
        code_size = len(vert_shader_code),
        code = raw_data(vert_shader_code),
        entrypoint = "main0",
        format = {.MSL},
        stage = .VERTEX,
        num_uniform_buffers = 1,
    })

    Vertex_Data :: struct {
        pos: Vec3,
        color: sdl.FColor,
        uv: [2]int,
        normal: Vec3,
    }
    WHITE :: sdl.FColor{1,1,1,1} 
    vertices := [dynamic]Vertex_Data{}
    for triangle in triangles {
        for i in 0..<len(triangle.p) {
            pos := triangle.p[i]
            uv: [2]int
            switch i {
            case 0: {}
            case 1: uv[0] = 1
            case 2: uv[1] = 1
            case: panic("unacceptable")
            }
            normal := calculate_triangle_normal(triangle)
            append_elem(&vertices, Vertex_Data {
                pos = pos,
                color = {1,1,1,1},
                uv = uv,
                normal = normal,
            })
        }
    }
    vertices_byte_size := len(vertices) * size_of(vertices[0])

    indices := [dynamic]u16{}
    for triangle_index in 0..<len(triangles) {
        for vertex_index in 0..<3 {
            append_elem(&indices, u16(vertex_index + triangle_index * 3))
        }
    }
    indices_byte_size := len(indices) * size_of(indices[0])

    fmt.println(len(vertices), len(indices))

    index_buf := sdl.CreateGPUBuffer(gpu, {
        usage = {.INDEX},
        size = u32(indices_byte_size),
    })

    vertex_buf := sdl.CreateGPUBuffer(gpu, {
        usage = {.VERTEX},
        size = u32(vertices_byte_size),
    })

    transfer_buf := sdl.CreateGPUTransferBuffer(gpu, {
        usage = .UPLOAD,
        size = u32(vertices_byte_size) + u32(indices_byte_size),
    })

    texture_size := cobblestone_texture.w * cobblestone_texture.h * 4
    tex_transfer_buf := sdl.CreateGPUTransferBuffer(gpu, {
        usage = .UPLOAD,
        size = u32(texture_size),
    })

    transfer_mem := transmute([^]byte)sdl.MapGPUTransferBuffer(gpu, transfer_buf, false)
    tex_transfer_mem := sdl.MapGPUTransferBuffer(gpu, tex_transfer_buf, false)
    mem.copy(transfer_mem, raw_data(vertices), vertices_byte_size)
    mem.copy(transfer_mem[vertices_byte_size:], raw_data(indices), indices_byte_size)
    mem.copy(tex_transfer_mem, cobblestone_texture.pixels, int(texture_size))
    sdl.UnmapGPUTransferBuffer(gpu, transfer_buf)
    sdl.UnmapGPUTransferBuffer(gpu, tex_transfer_buf)

    cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)
    copy_pass := sdl.BeginGPUCopyPass(cmd_buf)

    sdl.UploadToGPUBuffer(copy_pass,
        { transfer_buffer = transfer_buf },
        { buffer = vertex_buf, size = u32(vertices_byte_size) },
        false
    )
    sdl.UploadToGPUBuffer(copy_pass,
        { transfer_buffer = transfer_buf, offset = u32(vertices_byte_size) },
        { buffer = index_buf, size = u32(indices_byte_size) },
        false
    )

    sdl.UploadToGPUTexture(copy_pass,
        { transfer_buffer = tex_transfer_buf },
        {
            texture = texture,
            w = u32(cobblestone_texture.w),
            h = u32(cobblestone_texture.h),
            d = 1
        },
        false
    )

    sdl.EndGPUCopyPass(copy_pass)
    handle_err(sdl.SubmitGPUCommandBuffer(cmd_buf))

    sdl.ReleaseGPUTransferBuffer(gpu, transfer_buf)

    sampler := sdl.CreateGPUSampler(gpu, {})

    vertex_attributes := []sdl.GPUVertexAttribute{
        {
            location = 0,
            format = .FLOAT3,
            offset = u32(offset_of(Vertex_Data, pos)),
        },
        {
            location = 1,
            format = .FLOAT4,
            offset = u32(offset_of(Vertex_Data, color)),
        },
        {
            location = 2,
            format = .FLOAT2,
            offset = u32(offset_of(Vertex_Data, uv)),
        },
        {
            location = 3,
            format = .FLOAT3,
            offset = u32(offset_of(Vertex_Data, normal)),
        },
    }

    pipeline := sdl.CreateGPUGraphicsPipeline(gpu, {
        fragment_shader = frag_shader,
        vertex_shader = vert_shader,
        primitive_type = .TRIANGLELIST,
        target_info = {
            num_color_targets = 1,
            color_target_descriptions = &(sdl.GPUColorTargetDescription{
                format = sdl.GetGPUSwapchainTextureFormat(gpu, window)
            }),
            has_depth_stencil_target = true,
            depth_stencil_format = .D16_UNORM,
        },
        vertex_input_state = {
            num_vertex_buffers = 1,
            vertex_buffer_descriptions = &(sdl.GPUVertexBufferDescription{
                slot = 0,
                pitch = size_of(Vertex_Data),
            }),
            num_vertex_attributes = u32(len(vertex_attributes)),
            vertex_attributes = raw_data(vertex_attributes),
        },
        depth_stencil_state = {
            enable_depth_test = true,
            enable_depth_write = true,
            compare_op = .LESS,
        },
    })

    // I should be fine to release them after they've been loaded in the pipeline.
    sdl.ReleaseGPUShader(gpu, frag_shader)
    sdl.ReleaseGPUShader(gpu, vert_shader)

    window_width: i32
    window_height: i32
    handle_err(sdl.GetWindowSize(window, &window_width, &window_height))
    aspect := f32(window_width) / f32(window_height)
    ROTATION_SPEED := linalg.to_radians(f32(90))
    rotation := f32(0)
    proj_mat := linalg.matrix4_perspective_f32(linalg.to_radians(f32(70)), aspect, 0.0001, 1000.)

    // I'm sending each matrix separately so the shaders can do lighting stuff
    UBO :: struct #max_field_align(16) {
        projection: matrix[4, 4]f32,
        view: matrix[4, 4]f32,
        model: matrix[4, 4]f32,
    }

    last_ticks := sdl.GetTicks()

    event: sdl.Event
    main_loop: for {
        state.mouse_move = {}
        new_ticks := sdl.GetTicks()
        delta := f32(new_ticks - last_ticks) / 1000
        last_ticks = new_ticks

        for sdl.PollEvent(&event) {
            #partial switch event.type {
            case .QUIT:
                break main_loop
            case .KEY_DOWN:
                if event.key.scancode == .ESCAPE {
                    break main_loop
                }
                state.key_down[event.key.scancode] = true
            case .KEY_UP:
                state.key_down[event.key.scancode] = false
            case .MOUSE_MOTION:
                state.mouse_move += {event.motion.xrel, event.motion.yrel}
            }
        }

        update_camera(&state, delta)

        cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)

        swap_texture: ^sdl.GPUTexture
        width: u32
        height: u32
        handle_err(sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swap_texture, &width, &height))

        // rotation += ROTATION_SPEED * delta
        model_mat := linalg.matrix4_translate_f32({0, 0, -10}) + linalg.matrix4_rotate_f32(rotation, {0, 1, 0})
        view_mat := linalg.matrix4_look_at_f32(state.camera.pos, state.camera.target, {0,1,0})

        ubo := UBO{
            projection = proj_mat,
            view = view_mat,
            model = model_mat,
        }

        ShaderConsts :: struct #max_field_align(16) {
            light_pos: Vec3,
            view_pos: Vec3,
        }
        consts := ShaderConsts {
            light_pos = {20,20,0},
            view_pos = state.camera.pos,
        }

        if swap_texture == nil {
            handle_err(sdl.SubmitGPUCommandBuffer(cmd_buf))
            continue
        }

        color_target_info := sdl.GPUColorTargetInfo{
            clear_color = {0.9, 0.9, 0.9, 1},
            load_op = .CLEAR,
            store_op = .STORE,
            texture = swap_texture,
        }

        depth_target_info := sdl.GPUDepthStencilTargetInfo{
            texture = depth_texture,
            load_op = .CLEAR,
            clear_depth = 1,
            store_op = .DONT_CARE,
        }

        render_pass := sdl.BeginGPURenderPass(
             cmd_buf,
             &color_target_info,
             1,
             &depth_target_info
        )

        sdl.BindGPUGraphicsPipeline(render_pass, pipeline)
        sdl.BindGPUVertexBuffers(render_pass, 0, &(sdl.GPUBufferBinding{
            buffer = vertex_buf,
            offset = 0,
        }), 1)

        sdl.PushGPUVertexUniformData(cmd_buf, 0, &ubo, size_of(ubo))
        sdl.PushGPUFragmentUniformData(cmd_buf, 0, &consts, size_of(consts))

        sdl.BindGPUIndexBuffer(render_pass, { buffer = index_buf }, ._16BIT)
        sdl.BindGPUFragmentSamplers(
            render_pass,
            0,
            &(sdl.GPUTextureSamplerBinding{
                texture = texture,
                sampler = sampler
            }),
            1
        )
        // sdl.DrawGPUPrimitives(render_pass, 6, 1, 0, 0)
        sdl.DrawGPUIndexedPrimitives(render_pass, u32(len(indices)), 1, 0, 0, 0)
        sdl.EndGPURenderPass(render_pass)

        handle_err(sdl.SubmitGPUCommandBuffer(cmd_buf))
    }
}
