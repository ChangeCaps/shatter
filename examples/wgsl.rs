use std::time::Instant;

use shatter::*;

wgsl! {
    struct Particle {
        position: vec2<f32>;
        velocity: vec2<f32>;
        radius: f32;
    };

    [[block]]
    struct Particles {
        particles: array<Particle>;
    };

    [[group(0), binding(0)]]
    var<storage, read_write> particles: Particles;

    [[block]]
    struct Uniforms {
        simulation_speed: f32; 
    };

    [[group(0), binding(1)]]
    var<uniform> uniforms: Uniforms;

    [[stage(compute), workgroup_size(1024, 1, 1)]]
    fn comp([[builtin(global_invocation_id)]] param: vec3<u32>) {
        let particle = &particles.particles[param.x];

        (*particle).position = (*particle).position + uniforms.simulation_speed;
    }
}

fn main() {
    let mut particles: Buffer<Particles> = Buffer::new();
    let mut uniforms = Buffer::<Uniforms>::new(); 

    uniforms.simulation_speed = 0.0;

    for _ in 0..1_000_000 {
        particles.push(Particle {
            position: Vec2::new(0.0, 0.0),
            velocity: Vec2::new(0.0, 0.0),
            radius: 5.0,
        });
    }

    let bindings = comp::Bindings {
        particles: &mut particles,
        uniforms: &uniforms,
    };

    comp(
        bindings,
        Dispatch::new(1_000_000 / comp::WORK_GROUP_SIZE.x, 1, 1),
    );

    uniforms.simulation_speed = 2.0;

    let t = Instant::now();

    let bindings = comp::Bindings {
        particles: &mut particles,
        uniforms: &uniforms,
    };

    comp::build(bindings)
        .dispatch_multiple(&[Dispatch::new(1_000_000 / comp::WORK_GROUP_SIZE.x, 1, 1); 100]);

    println!("{:?}", Instant::now() - t);

    let t = Instant::now();

    println!("{:?}", &particles.particles[0]);

    println!("{:?}", Instant::now() - t);
}
