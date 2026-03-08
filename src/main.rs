use pollster::block_on;
use wgpu::{BindGroupDescriptor, BindGroupEntry, BlasBuildEntry, BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, ExperimentalFeatures, Features, Limits, Queue, RequestAdapterOptions, TlasInstance, include_wgsl, util::{BufferInitDescriptor, DeviceExt}, wgt::{AccelerationStructureFlags, AccelerationStructureGeometryFlags, BufferDescriptor, CommandEncoderDescriptor, CreateBlasDescriptor, CreateTlasDescriptor, DeviceDescriptor}};
use winit::window::WindowAttributes;

fn main() {
    // test w/o event loop
    run_test();

    let event_loop = winit::event_loop::EventLoop::new().unwrap();

    // test with event loop, but not window
    run_test();

    let mut app = App;

    event_loop.run_app(&mut app).unwrap();
}

struct App;

impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        run_test();

        let _window = event_loop.create_window(WindowAttributes::default()).unwrap();

        run_test();
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
        event_loop.exit();
    }
}

fn run_test() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

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

    exec_case(&device, &queue, false, false, &pipeline);
    exec_case(&device, &queue, false, true, &pipeline);
    exec_case(&device, &queue, true, false, &pipeline);
    exec_case(&device, &queue, true, true, &pipeline);
}

fn exec_case(device: &Device, queue: &Queue, block_blas: bool, block_tlas: bool, pipeline: &ComputePipeline) {
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

    assert_eq!(res, 1);
}