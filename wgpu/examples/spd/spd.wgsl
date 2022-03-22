let SPD_LINEAR_SAMPLER = true;

struct StorageBuffer {
    data: array<vec4<f32>>;
};

struct SpdGlobalAtomicBuffer {
    counter: atomic<u32>;
};

struct SpdConstants {
    mips: u32;
    numWorkGroups: u32;
    invInputSize: vec2<f32>;
};

[[group(0), binding(0)]] var imgSrc: texture_2d<f32>;
[[group(0), binding(1)]] var imgDst0: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(2)]] var imgDst1: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(3)]] var imgDst2: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(4)]] var imgDst3: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(5)]] var imgDst4: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(6)]] var imgDst5: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(7)]] var srcSampler: sampler;
[[group(0), binding(8)]] var<storage, read_write> imgDst5Buffer: StorageBuffer;
[[group(0), binding(9)]] var<storage, read_write> spdGlobalAtomic: SpdGlobalAtomicBuffer;
[[group(0), binding(10)]] var<uniform> spdConstants: SpdConstants;
[[group(0), binding(11)]] var imgDst6: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(12)]] var imgDst7: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(13)]] var imgDst8: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(14)]] var imgDst9: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(15)]] var imgDst10: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(16)]] var imgDst11: texture_storage_2d<rgba8unorm,write>;

var<workgroup> spdIntermediate: array<array<vec4<f32>, 16>, 16>;
var<workgroup> spdCounter: u32;

fn SpdLoadSourceImage(p: vec2<u32>, slice: u32) -> vec4<f32> {
    if (SPD_LINEAR_SAMPLER) {
        let textureCoord = vec2<f32>(p) * spdConstants.invInputSize + spdConstants.invInputSize;
        return textureSampleLevel(imgSrc, srcSampler, textureCoord, 0.);
    }
    return textureLoad(imgSrc, vec2<i32>(p), 0);
}

fn SpdStore(p: vec2<u32>, value: vec4<f32>, mip: u32, slice: u32) {
    if (mip == 0u) {
        textureStore(imgDst0, vec2<i32>(p), value);
    } else if (mip == 1u) {
        textureStore(imgDst1, vec2<i32>(p), value);
    } else if (mip == 2u) {
        textureStore(imgDst2, vec2<i32>(p), value);
    } else if (mip == 3u) {
        textureStore(imgDst3, vec2<i32>(p), value);
    } else if (mip == 4u) {
        textureStore(imgDst4, vec2<i32>(p), value);
    } else if (mip == 5u) {
        textureStore(imgDst5, vec2<i32>(p), value);
        imgDst5Buffer.data[p.x + p.y * 64u] = value;
    } else if (mip == 6u) {
        textureStore(imgDst6, vec2<i32>(p), value);
    } else if (mip == 7u) {
        textureStore(imgDst7, vec2<i32>(p), value);
    } else if (mip == 8u) {
        textureStore(imgDst8, vec2<i32>(p), value);
    } else if (mip == 9u) {
        textureStore(imgDst9, vec2<i32>(p), value);
    } else if (mip == 10u) {
        //textureStore(imgDst10, vec2<i32>(p), value);
    } else if (mip == 11u) {
        //textureStore(imgDst11, vec2<i32>(p), value);
    }
}

fn SpdLoad(p: vec2<u32>, slice: u32) -> vec4<f32> {
    // return textureLoad(imgDst5, vec2<i32>(p), 0);
    return imgDst5Buffer.data[p.x + p.y * 64u];
}

fn SpdIncreaseAtomicCounter(slice: u32) {
    spdCounter = atomicAdd(&spdGlobalAtomic.counter, 1u);
}

fn SpdGetAtomicCounter() -> u32 {
    return spdCounter;
}

fn SpdResetAtomicCounter(slice: u32) {
    atomicStore(&spdGlobalAtomic.counter, 0u);
}

fn SpdLoadIntermediate(x: u32, y: u32) -> vec4<f32> {
    return spdIntermediate[x][y];
}

fn SpdStoreIntermediate(x: u32, y: u32, value: vec4<f32>) {
    spdIntermediate[x][y] = value;
}

fn SpdReduce4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    return (v0+v1+v2+v3)*0.25;
}

fn SpdWorkgroupShuffleBarrier() {
    workgroupBarrier();
}

fn SpdExitWorkgroup(numWorkGroups: u32, localInvocationIndex: u32, slice: u32) -> bool
{
    if (localInvocationIndex == 0u) {
        SpdIncreaseAtomicCounter(slice);
    }
    SpdWorkgroupShuffleBarrier();
    return (SpdGetAtomicCounter() != (numWorkGroups - 1u));
}

fn SpdReduceIntermediate(i0: vec2<u32>, i1: vec2<u32>, i2: vec2<u32>, i3: vec2<u32>) -> vec4<f32> {
    let v0 = SpdLoadIntermediate(i0.x, i0.y);
    let v1 = SpdLoadIntermediate(i1.x, i1.y);
    let v2 = SpdLoadIntermediate(i2.x, i2.y);
    let v3 = SpdLoadIntermediate(i3.x, i3.y);
    return SpdReduce4(v0, v1, v2, v3);
}

fn SpdReduceLoad4_(i0: vec2<u32>, i1: vec2<u32>, i2: vec2<u32>, i3: vec2<u32>, slice: u32) -> vec4<f32>
{
    let v0 = SpdLoad(i0, slice);
    let v1 = SpdLoad(i1, slice);
    let v2 = SpdLoad(i2, slice);
    let v3 = SpdLoad(i3, slice);
    return SpdReduce4(v0, v1, v2, v3);
}

fn SpdReduceLoad4(base: vec2<u32>, slice: u32) -> vec4<f32>
{
    return SpdReduceLoad4_(
        vec2<u32>(base + vec2<u32>(0u, 0u)),
        vec2<u32>(base + vec2<u32>(0u, 1u)),
        vec2<u32>(base + vec2<u32>(1u, 0u)),
        vec2<u32>(base + vec2<u32>(1u, 1u)),
        slice);
}

fn SpdReduceLoadSourceImage4(i0: vec2<u32>, i1: vec2<u32>, i2: vec2<u32>, i3: vec2<u32>, slice: u32) -> vec4<f32>
{
    let v0 = SpdLoadSourceImage(i0, slice);
    let v1 = SpdLoadSourceImage(i1, slice);
    let v2 = SpdLoadSourceImage(i2, slice);
    let v3 = SpdLoadSourceImage(i3, slice);
    return SpdReduce4(v0, v1, v2, v3);
}

fn SpdReduceLoadSourceImage(base: vec2<u32>, slice: u32) -> vec4<f32>
{
    if (SPD_LINEAR_SAMPLER) {
        return SpdLoadSourceImage(base, slice);
    }
    return SpdReduceLoadSourceImage4(
        vec2<u32>(base + vec2<u32>(0u, 0u)),
        vec2<u32>(base + vec2<u32>(0u, 1u)),
        vec2<u32>(base + vec2<u32>(1u, 0u)),
        vec2<u32>(base + vec2<u32>(1u, 1u)),
        slice);
}

fn SpdDownsampleMips_0_1_LDS(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    var v: array<vec4<f32>, 4>;

    var tex = workGroupID.xy * 64u + vec2<u32>(x * 2u, y * 2u);
    var pix = workGroupID.xy * 32u + vec2<u32>(x, y);
    v[0] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[0], 0u, slice);

    tex = workGroupID.xy * 64u + vec2<u32>(x * 2u + 32u, y * 2u);
    pix = workGroupID.xy * 32u + vec2<u32>(x + 16u, y);
    v[1] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[1], 0u, slice);

    tex = workGroupID.xy * 64u + vec2<u32>(x * 2u, y * 2u + 32u);
    pix = workGroupID.xy * 32u + vec2<u32>(x, y + 16u);
    v[2] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[2], 0u, slice);

    tex = workGroupID.xy * 64u + vec2<u32>(x * 2u + 32u, y * 2u + 32u);
    pix = workGroupID.xy * 32u + vec2<u32>(x + 16u, y + 16u);
    v[3] = SpdReduceLoadSourceImage(tex, slice);
    SpdStore(pix, v[3], 0u, slice);

    if (mip <= 1u) {
        return;
    }

    for (var i = 0; i < 4; i = i + 1)
    {
        SpdStoreIntermediate(x, y, v[i]);
        SpdWorkgroupShuffleBarrier();
        if (localInvocationIndex < 64u)
        {
            v[i] = SpdReduceIntermediate(
                vec2<u32>(x * 2u + 0u, y * 2u + 0u),
                vec2<u32>(x * 2u + 1u, y * 2u + 0u),
                vec2<u32>(x * 2u + 0u, y * 2u + 1u),
                vec2<u32>(x * 2u + 1u, y * 2u + 1u)
            );
            SpdStore(workGroupID.xy * 16u + vec2<u32>(x + u32(i % 2) * 8u, y + u32(i / 2) * 8u), v[i], 1u, slice);
        }
        SpdWorkgroupShuffleBarrier();
    }

    if (localInvocationIndex < 64u)
    {
        SpdStoreIntermediate(x + 0u, y + 0u, v[0]);
        SpdStoreIntermediate(x + 8u, y + 0u, v[1]);
        SpdStoreIntermediate(x + 0u, y + 8u, v[2]);
        SpdStoreIntermediate(x + 8u, y + 8u, v[3]);
    }
}

fn SpdDownsampleMips_0_1(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    SpdDownsampleMips_0_1_LDS(x, y, workGroupID, localInvocationIndex, mip, slice);
}

fn SpdDownsampleMip_2(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    if (localInvocationIndex < 64u)
    {
        let v = SpdReduceIntermediate(
            vec2<u32>(x * 2u + 0u, y * 2u + 0u),
            vec2<u32>(x * 2u + 1u, y * 2u + 0u),
            vec2<u32>(x * 2u + 0u, y * 2u + 1u),
            vec2<u32>(x * 2u + 1u, y * 2u + 1u)
        );
        SpdStore(workGroupID.xy * 8u + vec2<u32>(x, y), v, mip, slice);
        // store to LDS, try to reduce bank conflicts
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // ...
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        SpdStoreIntermediate(x * 2u + y % 2u, y * 2u, v);
    }
}

fn SpdDownsampleMip_3(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    if (localInvocationIndex < 16u)
    {
        // x 0 x 0
        // 0 0 0 0
        // 0 x 0 x
        // 0 0 0 0
        let v = SpdReduceIntermediate(
            vec2<u32>(x * 4u + 0u + 0u, y * 4u + 0u),
            vec2<u32>(x * 4u + 2u + 0u, y * 4u + 0u),
            vec2<u32>(x * 4u + 0u + 1u, y * 4u + 2u),
            vec2<u32>(x * 4u + 2u + 1u, y * 4u + 2u)
        );
        SpdStore(workGroupID.xy * 4u + vec2<u32>(x, y), v, mip, slice);
        // store to LDS
        // x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
        // ...
        // 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
        // ...
        // 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
        // ...
        SpdStoreIntermediate(x * 4u + y, y * 4u, v);
    }
}

fn SpdDownsampleMip_4(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    if (localInvocationIndex < 4u)
    {
        // x 0 0 0 x 0 0 0
        // ...
        // 0 x 0 0 0 x 0 0
        let v = SpdReduceIntermediate(
            vec2<u32>(x * 8u + 0u + 0u + y * 2u, y * 8u + 0u),
            vec2<u32>(x * 8u + 4u + 0u + y * 2u, y * 8u + 0u),
            vec2<u32>(x * 8u + 0u + 1u + y * 2u, y * 8u + 4u),
            vec2<u32>(x * 8u + 4u + 1u + y * 2u, y * 8u + 4u)
        );
        SpdStore(workGroupID.xy * 2u + vec2<u32>(x, y), v, mip, slice);
        // store to LDS
        // x x x x 0 ...
        // 0 ...
        SpdStoreIntermediate(x + y * 2u, 0u, v);
    }
}

fn SpdDownsampleMip_5(workGroupID: vec2<u32>, localInvocationIndex: u32, mip: u32, slice: u32)
{
    if (localInvocationIndex < 1u)
    {
        // x x x x 0 ...
        // 0 ...
        let v = SpdReduceIntermediate(
            vec2<u32>(0u, 0u),
            vec2<u32>(1u, 0u),
            vec2<u32>(2u, 0u),
            vec2<u32>(3u, 0u)
        );
        SpdStore(workGroupID.xy, v, mip, slice);
    }
}

fn SpdDownsampleMips_6_7(x: u32, y: u32, mips: u32, slice: u32)
{
    var tex = vec2<u32>(x * 4u + 0u, y * 4u + 0u);
    var pix = vec2<u32>(x * 2u + 0u, y * 2u + 0u);
    let v0 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v0, 6u, slice);

    tex = vec2<u32>(x * 4u + 2u, y * 4u + 0u);
    pix = vec2<u32>(x * 2u + 1u, y * 2u + 0u);
    let v1 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v1, 6u, slice);

    tex = vec2<u32>(x * 4u + 0u, y * 4u + 2u);
    pix = vec2<u32>(x * 2u + 0u, y * 2u + 1u);
    let v2 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v2, 6u, slice);

    tex = vec2<u32>(x * 4u + 2u, y * 4u + 2u);
    pix = vec2<u32>(x * 2u + 1u, y * 2u + 1u);
    let v3 = SpdReduceLoad4(tex, slice);
    SpdStore(pix, v3, 6u, slice);

    if (mips <= 7u) { return; }
    // no barrier needed, working on values only from the same thread

    let v = SpdReduce4(v0, v1, v2, v3);
    SpdStore(vec2<u32>(x, y), v, 7u, slice);
    SpdStoreIntermediate(x, y, v);
}

fn SpdDownsampleNextFour(x: u32, y: u32, workGroupID: vec2<u32>, localInvocationIndex: u32, baseMip: u32, mips: u32, slice: u32)
{
    if (mips <= baseMip) { return; }
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_2(x, y, workGroupID, localInvocationIndex, baseMip, slice);

    if (mips <= baseMip + 1u) { return; }
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_3(x, y, workGroupID, localInvocationIndex, baseMip + 1u, slice);

    if (mips <= baseMip + 2u) { return; }
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_4(x, y, workGroupID, localInvocationIndex, baseMip + 2u, slice);

    if (mips <= baseMip + 3u) { return; }
    SpdWorkgroupShuffleBarrier();
    SpdDownsampleMip_5(workGroupID, localInvocationIndex, baseMip + 3u, slice);
}

fn ABfe(src: u32, off: u32, bits: u32) -> u32 {
    let mask = (1u<<bits) - 1u;
    return (src>>off)&mask;
}

fn ABfi(src: u32, ins: u32, mask: u32) -> u32 {
    return (ins&mask)|(src&(~mask));
}

fn ABfiM(src: u32, ins: u32, bits: u32) -> u32 {
    let mask = (1u<<bits) - 1u;
    return (ins&mask)|(src&(~mask));
}

// Simple remap 64x1 to 8x8 with rotated 2x2 pixel quads in quad linear.
//  543210
//  ======
//  ..xxx.
//  yy...y
fn ARmp8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(ABfe(a,1u,3u),ABfiM(ABfe(a,3u,3u),a,1u));
}

// More complex remap 64x1 to 8x8 which is necessary for 2D wave reductions.
//  543210
//  ======
//  .xx..x
//  y..yy.
// Details,
//  LANE TO 8x8 MAPPING
//  ===================
//  00 01 08 09 10 11 18 19
//  02 03 0a 0b 12 13 1a 1b
//  04 05 0c 0d 14 15 1c 1d
//  06 07 0e 0f 16 17 1e 1f
//  20 21 28 29 30 31 38 39
//  22 23 2a 2b 32 33 3a 3b
//  24 25 2c 2d 34 35 3c 3d
//  26 27 2e 2f 36 37 3e 3f
fn ARmpRed8x8(a: u32) -> vec2<u32> {
    return vec2<u32>(ABfiM(ABfe(a,2u,3u),a,1u),ABfiM(ABfe(a,3u,3u),ABfe(a,1u,2u),2u));
}

fn SpdDownsample(
    workGroupID: vec2<u32>,
    localInvocationIndex: u32,
    mips: u32,
    numWorkGroups: u32,
    slice: u32
) {
    let sub_xy = ARmpRed8x8(localInvocationIndex % 64u);
    let x = sub_xy.x + 8u * ((localInvocationIndex >> 6u) % 2u);
    let y = sub_xy.y + 8u * ((localInvocationIndex >> 7u));
    SpdDownsampleMips_0_1(x, y, workGroupID, localInvocationIndex, mips, slice);

    SpdDownsampleNextFour(x, y, workGroupID, localInvocationIndex, 2u, mips, slice);

    if (mips <= 6u) { return; }

    if (SpdExitWorkgroup(numWorkGroups, localInvocationIndex, slice)) { return; }

    SpdResetAtomicCounter(slice);

    // After mip 6 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
    SpdDownsampleMips_6_7(x, y, mips, slice);

    SpdDownsampleNextFour(x, y, vec2<u32>(0u,0u), localInvocationIndex, 8u, mips, slice);
}

[[stage(compute), workgroup_size(256)]]
fn main(
    [[builtin(workgroup_id)]] workgroup_id: vec3<u32>,
    [[builtin(local_invocation_index)]] local_invocation_index: u32
) {
    SpdDownsample(workgroup_id.xy, local_invocation_index, spdConstants.mips, spdConstants.numWorkGroups, workgroup_id.z);
}
