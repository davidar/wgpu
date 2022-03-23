#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, mem, num::NonZeroU32};
use wgpu::util::DeviceExt;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const MIP_LEVEL_COUNT: u32 = 11;
const MIP_PASS_COUNT: u32 = MIP_LEVEL_COUNT - 1;

fn create_texels(size: usize, cx: f32, cy: f32) -> Vec<u8> {
    use std::iter;

    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let mut x = 4.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let mut y = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let mut count = 0;
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 2) as u8)
                .chain(iter::once(0xFF - (count * 5) as u8))
                .chain(iter::once(0xFF - (count * 13) as u8))
                .chain(iter::once(std::u8::MAX))
        })
        .collect()
}

struct QuerySets {
    timestamp: wgpu::QuerySet,
    timestamp_period: f32,
    pipeline_statistics: wgpu::QuerySet,
    data_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

type TimestampQueries = [TimestampData; 1 as usize];
type PipelineStatisticsQueries = [u64; 1 as usize];

fn pipeline_statistics_offset() -> wgpu::BufferAddress {
    (mem::size_of::<TimestampQueries>() as wgpu::BufferAddress)
        .max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

struct Example {
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    draw_pipeline: wgpu::RenderPipeline,
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 1000.0);
        let mx_view = cgmath::Matrix4::look_at_rh(
            cgmath::Point3::new(0f32, 0.0, 10.0),
            cgmath::Point3::new(0f32, 50.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_correction = framework::OPENGL_TO_WGPU_MATRIX;
        mx_correction * mx_projection * mx_view
    }

    fn generate_mipmaps(
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        query_sets: &Option<QuerySets>,
        mip_count: u32,
    ) {
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("spd.wgsl"))),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spd"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("mip"),
                    format: None,
                    dimension: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: NonZeroU32::new(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                })
            })
            .collect::<Vec<_>>();

        let num_workgroups_per_dimension = 1 << (MIP_LEVEL_COUNT - 6);
        let image_size = (1 << MIP_LEVEL_COUNT) as f32;
        let inv_image_size = 1. / image_size;

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4 * 4 * 64 * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let global_atomic_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let constants = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&[
                MIP_PASS_COUNT,
                num_workgroups_per_dimension * num_workgroups_per_dimension,
                inv_image_size.to_bits(), inv_image_size.to_bits(),
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let start_time = std::time::Instant::now();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&views[5]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&views[6]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: storage_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: global_atomic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: constants.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(&views[7]),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(&views[8]),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&views[9]),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&views[10]),
                },
                /*wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&views[11]),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&views[12]),
                },*/
            ],
            label: None,
        });

        {
            let mut rpass = encoder.begin_compute_pass(&Default::default());
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, 0);
            }
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.dispatch(num_workgroups_per_dimension, num_workgroups_per_dimension, 1);
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, 1);
            }
        }

        println!("Total time: {:.3} ms", start_time.elapsed().as_micros() as f32 / 1e3);

        if let Some(ref query_sets) = query_sets {
            encoder.resolve_query_set(
                &query_sets.timestamp,
                0..2,
                &query_sets.data_buffer,
                0,
            );
        }
    }
}

impl framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PIPELINE_STATISTICS_QUERY
    }

    fn required_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_textures_per_shader_stage: MIP_PASS_COUNT,
            ..wgpu::Limits::downlevel_defaults()
        }
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Create the texture
        let size = 1 << MIP_LEVEL_COUNT;
        let texels = create_texels(size as usize, -0.8, 0.156);
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: MIP_LEVEL_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            label: None,
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        //Note: we could use queue.write_texture instead, and this is what other
        // examples do, but here we want to show another way to do this.
        let temp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Temporary Buffer"),
            contents: texels.as_slice(),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        init_encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &temp_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(4 * size).unwrap()),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            texture_extent,
        );

        // Create other resources
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create the render pipeline
        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../mipmap/draw.wgsl"))),
        });

        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[config.format.into()],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create bind group
        let bind_group_layout = draw_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        // If both kinds of query are supported, use queries
        let query_sets = if device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::PIPELINE_STATISTICS_QUERY)
        {
            // For N total mips, it takes N - 1 passes to generate them, and we're measuring those.
            let mip_passes = MIP_LEVEL_COUNT - 1;

            // Create the timestamp query set. We need twice as many queries as we have passes,
            // as we need a query at the beginning and at the end of the operation.
            let timestamp = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: mip_passes * 2,
                ty: wgpu::QueryType::Timestamp,
            });
            // Timestamp queries use an device-specific timestamp unit. We need to figure out how many
            // nanoseconds go by for the timestamp to be incremented by one. The period is this value.
            let timestamp_period = queue.get_timestamp_period();

            // We only need one pipeline statistics query per pass.
            let pipeline_statistics = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: mip_passes,
                ty: wgpu::QueryType::PipelineStatistics(
                    wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                ),
            });

            // This databuffer has to store all of the query results, 2 * passes timestamp queries
            // and 1 * passes statistics queries. Each query returns a u64 value.
            let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query buffer"),
                size: pipeline_statistics_offset()
                    + mem::size_of::<PipelineStatisticsQueries>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Some(QuerySets {
                timestamp,
                timestamp_period,
                pipeline_statistics,
                data_buffer,
            })
        } else {
            None
        };

        Self::generate_mipmaps(
            &mut init_encoder,
            device,
            &texture,
            &query_sets,
            MIP_LEVEL_COUNT,
        );

        queue.submit(Some(init_encoder.finish()));
        if let Some(ref query_sets) = query_sets {
            // We can ignore the future as we're about to wait for the device.
            let _ = query_sets
                .data_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read);
            // Wait for device to be done rendering mipmaps
            device.poll(wgpu::Maintain::Wait);
            // This is guaranteed to be ready.
            let timestamp_view = query_sets
                .data_buffer
                .slice(..mem::size_of::<TimestampQueries>() as wgpu::BufferAddress)
                .get_mapped_range();
            let pipeline_stats_view = query_sets
                .data_buffer
                .slice(pipeline_statistics_offset()..)
                .get_mapped_range();
            // Convert the raw data into a useful structure
            let timestamp_data: &TimestampQueries = bytemuck::from_bytes(&*timestamp_view);
            let pipeline_stats_data: &PipelineStatisticsQueries =
                bytemuck::from_bytes(&*pipeline_stats_view);
            // Iterate over the data
            for (_idx, (timestamp, _pipeline)) in timestamp_data
                .iter()
                .zip(pipeline_stats_data.iter())
                .enumerate()
            {
                // Figure out the timestamp differences and multiply by the period to get nanoseconds
                let nanoseconds =
                    (timestamp.end - timestamp.start) as f32 * query_sets.timestamp_period;
                // Nanoseconds is a bit small, so lets use microseconds.
                let microseconds = nanoseconds / 1000.0;
                // Print the data!
                println!(
                    "Generating {} mip levels took {:.3} μs",
                    MIP_PASS_COUNT,
                    microseconds,
                );
            }
        }

        Example {
            bind_group,
            uniform_buf,
            draw_pipeline,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let clear_color = wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&self.draw_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("mipmap");
}

#[test]
fn mipmap() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/mipmap/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .backend_failure(wgpu::Backends::GL),
        tolerance: 50,
        max_outliers: 5000, // Mipmap sampling is highly variant between impls. This is currently bounded by lavapipe
    });
}