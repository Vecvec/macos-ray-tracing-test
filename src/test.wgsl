enable wgpu_ray_query;

@group(0) @binding(0)
var acc: acceleration_structure;

@group(0) @binding(1)
var<storage, read_write> has_hit: u32;

@compute
@workgroup_size(1)
fn test() {
    var rq:ray_query;

    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xff, 0.0, 10.0, vec3(0.0), vec3(0.0, 1.0, 0.0)));

    while (rayQueryProceed(&rq)) {}

    let intersection = rayQueryGetCommittedIntersection(&rq);

    has_hit = intersection.kind;
}