use std::{fs, path::Path};

use shatter::*;

// inspired by:
// https://www.shadertoy.com/view/4df3Rn
wgsl! {
    [[group(0), binding(0)]]
    var texture: texture_storage_2d<rgba8unorm, write>;

    [[block]]
    struct Uniforms {
        position: vec2<f32>;
        zoom: f32;
    };

    [[group(0), binding(1)]]
    var<uniform> uniforms: Uniforms;

    let SCALE = 4.0;
    let AA = 3;

    [[stage(compute), workgroup_size(8, 8, 1)]]
    fn mandelbrot([[builtin(global_invocation_id)]] param: vec3<u32>) {
        var color = vec3<f32>(0.0);

        for (var m = 0; m < AA; m = m + 1) {
            let x_offset = f32(m) / f32(AA) - 0.5;

            for (var n = 0; n < AA; n = n + 1) {
                let y_offset = f32(n) / f32(AA) - 0.5;

                let size = textureDimensions(texture);

                var x = (f32(param.x) + x_offset) / f32(size.x) * SCALE - SCALE / 2.0;
                var y = (f32(param.y) + y_offset) / f32(size.y) * SCALE - SCALE / 2.0;

                x = x / uniforms.zoom - uniforms.position.x;
                y = y / uniforms.zoom - uniforms.position.y;

                var l = 0.0;
                var z = vec2<f32>(0.0);
                for (var i = 0; i < 512; i = i + 1) {
                    z = vec2<f32>(
                        z.x * z.x - z.y * z.y + x,
                        z.y * z.x + z.x * z.y + y,
                    );

                    if (dot(z, z) > pow(256.0, 2.0)) {
                        break;
                    }

                    l = l + 1.0;
                }

                if (l > 511.0) {
                    l = 0.0;
                }

                let smooth = l - log2(log2(dot(z, z))) + 4.0;

                let sub_color = 0.5 + 0.5 * cos(3.0 + smooth * 0.15 + vec3<f32>(0.0, 0.6, 1.0));
                color = color + sub_color;
            }
        }

        color = color / f32(AA * AA);

        let out_color = vec4<f32>(color, 1.0);

        textureStore(texture, vec2<i32>(param.xy), out_color);
    }
}

macro_rules! seg {
    (
        $msg:expr,
        $($tt:tt)*
    ) => {
        let t = std::time::Instant::now();
        $($tt)*
        println!("{}: {:?}", $msg, std::time::Instant::now() - t);
    };
}

fn main() {
    seg! {
        "initialize",
        Instance::global();
    }

    seg! {
        "create texture",
        let mut texture = Texture2d::<Rgba8Unorm>::new(256, 256);
    }

    seg! {
        "create uniforms",
        let mut uniforms = Buffer::<Uniforms>::new();
    }

    uniforms.position = Vec2::new(0.745, 0.186);

    let dispatch = Dispatch::new(
        texture.width() as u32 / mandelbrot::WORK_GROUP_SIZE.x,
        texture.height() as u32 / mandelbrot::WORK_GROUP_SIZE.y,
        1,
    );

    if !Path::new("images").exists() {
        fs::create_dir("images").unwrap();
    }

    let file = fs::File::create("images/mandelbrot.gif").unwrap();
    let mut encoder =
        gif::Encoder::new(file, texture.width() as u16, texture.height() as u16, &[]).unwrap();

    encoder.set_repeat(gif::Repeat::Infinite).unwrap();

    let frames = 200u32;

    for frame in 0..frames {
        let zoom = frame as f32 * 0.02;

        uniforms.zoom = zoom.powi(8);

        let bindings = mandelbrot::Bindings {
            texture: &mut texture,
            uniforms: &uniforms,
        };

        seg! {
            "dispatch",
            mandelbrot(bindings, dispatch);
            let mut bytes = texture.bytes().to_vec();
        }

        seg! {
            "encode frame",
            let frame =
                gif::Frame::from_rgba(texture.width() as u16, texture.height() as u16, &mut bytes);

            encoder.write_frame(&frame).unwrap();
        }
    }
}
