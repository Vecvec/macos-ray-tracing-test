use pollster::block_on;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BlasBuildEntry, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, ExperimentalFeatures, Features, Limits, Queue, RequestAdapterOptions, TlasInstance, include_wgsl, util::{BufferInitDescriptor, DeviceExt}, wgt::{AccelerationStructureFlags, AccelerationStructureGeometryFlags, BufferDescriptor, CommandEncoderDescriptor, CreateBlasDescriptor, CreateTlasDescriptor, DeviceDescriptor}};

fn main() {
    loop {
        let mut tests = Tests::default();

        run_test(&mut tests, "");

        tests.assert_success();
    }
}

#[test]
fn test() {
    loop {
        let mut tests = Tests::default();

        run_test(&mut tests, "");

        tests.assert_success();
    }
}

#[derive(Default)]
struct Tests {
    all: Vec<String>,
    total_failure: bool,
}

impl Tests {
    fn add(&mut self, name: &str, block_blas: bool, block_tlas: bool, ty: u32) {
        let success = ty == 1;

        self.total_failure = self.total_failure || !success;

        self.all.push(format!("{name} {block_blas} {block_tlas}: {} (ty: {ty})", if success { "succeeded" } else { "failed" }));
    }

    fn assert_success(&self) {
        for case in &self.all {
            println!("{case}");
        }

        assert!(!self.total_failure)
    }
}

fn run_test(cases: &mut Tests, name: &str) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default())).unwrap();

    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features: Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: Limits::defaults().using_minimum_supported_acceleration_structure_values(),
        experimental_features: unsafe {
            ExperimentalFeatures::enabled()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    })).unwrap();

    let module = device.create_shader_module(include_wgsl!("test.wgsl"));

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });

    exec_case(&device, &queue, false, false, &pipeline, cases, name);
    exec_case(&device, &queue, false, true, &pipeline, cases, name);
    exec_case(&device, &queue, true, false, &pipeline, cases, name);
    exec_case(&device, &queue, true, true, &pipeline, cases, name);
}

fn exec_case(device: &Device, queue: &Queue, block_blas: bool, block_tlas: bool, pipeline: &ComputePipeline, cases: &mut Tests, name: &str) {
    unsafe { device.start_graphics_debugger_capture(); }
    let tri_sizes = BlasTriangleGeometrySizeDescriptor { vertex_format: wgpu::VertexFormat::Float32x3, vertex_count: 3, index_format: None, index_count: None, flags: AccelerationStructureGeometryFlags::OPAQUE };
    let sizes = wgpu::wgt::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![tri_sizes.clone()] };

    let blas = device.create_blas(&CreateBlasDescriptor {
        label: None,
        flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::wgt::AccelerationStructureUpdateMode::Build,
    }, sizes.clone());

    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_BUILD,
        update_mode: wgpu::wgt::AccelerationStructureUpdateMode::Build,
    });

    tlas[0] = Some(TlasInstance::new(&blas, 
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0
            ], 0, 0xFF));

    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Blas vertex buf"),
        contents: bytemuck::cast_slice(&[[1.0f32, 1.0, 0.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]]),
        usage: BufferUsages::BLAS_INPUT,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("base encoder"),
    });

    encoder.build_acceleration_structures([&BlasBuildEntry {
        blas: &blas,
        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry { size: &tri_sizes, vertex_buffer: &buffer, first_vertex: 0, vertex_stride: 12, index_buffer: None, first_index: None, transform_buffer: None, transform_buffer_offset: None }]),
    }], []);

    if block_blas {
        queue.submit([encoder.finish()]);
        device.poll(wgpu::wgt::PollType::wait_indefinitely()).unwrap();
        encoder = device.create_command_encoder(&Default::default());
    }

    encoder.build_acceleration_structures([], [&tlas]);

    if block_tlas {
        queue.submit([encoder.finish()]);
        device.poll(wgpu::wgt::PollType::wait_indefinitely()).unwrap();
        encoder = device.create_command_encoder(&Default::default());
    }

    let out_buf = device.create_buffer(&BufferDescriptor {
        label: Some("shader output buffer"),
        size: size_of::<u32>() as _,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: tlas.as_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            }
        ],
    });

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    

    let read_back = device.create_buffer(&BufferDescriptor { label: Some("readback buffer"), size: out_buf.size(), usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });

    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_back, 0, out_buf.size());

    queue.submit([encoder.finish()]);
    read_back.map_async(wgpu::MapMode::Read, .., |res| res.unwrap());
    device.poll(wgpu::wgt::PollType::wait_indefinitely()).unwrap();

    let range = read_back.get_mapped_range(..);

    let res = u32::from_ne_bytes(*bytemuck::from_bytes(&range));

    drop(range);

    read_back.unmap();

    unsafe { device.stop_graphics_debugger_capture(); }

    cases.add(name, block_blas, block_tlas, res);
}