package main

import "core:fmt"
import sdl "vendor:sdl3"

handle_err :: proc(success: bool) {
    if !success {
        err := sdl.GetError()
        fmt.println(err)
        panic("quitting due to error")
    }
}

frag_shader_code := #load("./shaders/shader.msl.frag")
vert_shader_code := #load("./shaders/shader.msl.vert")

main :: proc() {
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

    frag_shader := sdl.CreateGPUShader(gpu, {
        code_size = len(frag_shader_code),
        code = raw_data(frag_shader_code),
        // the cross-compilation process turns main into main0
        entrypoint = "main0",
        format = {.MSL},
        stage = .FRAGMENT,
    })

    vert_shader := sdl.CreateGPUShader(gpu, {
        code_size = len(vert_shader_code),
        code = raw_data(vert_shader_code),
        entrypoint = "main0",
        format = {.MSL},
        stage = .VERTEX,
    })

    pipeline := sdl.CreateGPUGraphicsPipeline(gpu, {
        fragment_shader = frag_shader,
        vertex_shader = vert_shader,
        primitive_type = .TRIANGLELIST,
        target_info = {
            num_color_targets = 1,
            color_target_descriptions = &(sdl.GPUColorTargetDescription{
                format = sdl.GetGPUSwapchainTextureFormat(gpu, window)
            })
        },
    })

    // I should be fine to release them after they've been loaded in the pipeline.
    sdl.ReleaseGPUShader(gpu, frag_shader)
    sdl.ReleaseGPUShader(gpu, vert_shader)

    event: sdl.Event
    main_loop: for {
        for sdl.PollEvent(&event) {
            #partial switch event.type {
            case .QUIT:
                break main_loop
            }
        }

        cmd_buf := sdl.AcquireGPUCommandBuffer(gpu)

        swap_texture: ^sdl.GPUTexture
        width: u32
        height: u32
        handle_err(sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swap_texture, &width, &height))

        if swap_texture == nil {
            handle_err(sdl.SubmitGPUCommandBuffer(cmd_buf))
            continue
        }

        color_target_info := sdl.GPUColorTargetInfo{
            clear_color = {0.8, 0.5, 0.5, 1},
            load_op = .CLEAR,
            store_op = .STORE,
            texture = swap_texture,
        }

        render_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target_info, 1, nil)
        sdl.BindGPUGraphicsPipeline(render_pass, pipeline)
        sdl.DrawGPUPrimitives(render_pass, 3, 1, 0, 0)
        sdl.EndGPURenderPass(render_pass)

        handle_err(sdl.SubmitGPUCommandBuffer(cmd_buf))
    }
}
