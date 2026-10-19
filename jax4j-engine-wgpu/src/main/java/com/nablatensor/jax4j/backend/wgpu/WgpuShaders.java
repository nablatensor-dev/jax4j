/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.jax4j.backend.wgpu;

/**
 * Explicit WGSL for cooperative kernels: workgroup synchronization must never become subgroup-only.
 */
final class WgpuShaders {
  private WgpuShaders() {}

  static String source(String kernel) {
    return switch (kernel) {
      case "matmul" -> MATMUL;
      case "batched_matmul" -> BATCHED_MATMUL;
      case "reduce_sum", "reduce_sum_p1", "reduce_max", "reduce_max_p1" ->
          reduction(kernel.contains("max"), kernel.endsWith("p1"));
      default -> null;
    };
  }

  private static String reduction(boolean max, boolean phase1) {
    return """
    struct Params { n: u32, }
    @group(0) @binding(0) var<storage,read_write> output: array<f32>;
    @group(0) @binding(1) var<storage,read> input: array<f32>;
    @group(0) @binding(2) var<storage,read> p: Params;
    var<workgroup> data: array<f32,256>;
    @compute @workgroup_size(256)
    fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
      let tid=lid.x;
      var acc=INITIAL;
      for(var i=START;i<p.n;i+=STRIDE){ acc=COMBINE; }
      data[tid]=acc;
      workgroupBarrier();
      for(var stride=128u;stride>0u;stride>>=1u) {
        if(tid<stride){data[tid]=REDUCE;}
        workgroupBarrier();
      }
      if(tid==0u){output[DEST]=data[0];}
    }
    """
        .replace("INITIAL", max ? "-3.402823466e38" : "0.0")
        .replace("START", phase1 ? "gid.x" : "tid")
        .replace("STRIDE", phase1 ? "groups.x*256u" : "256u")
        .replace("COMBINE", max ? "max(acc,input[i])" : "acc+input[i]")
        .replace("REDUCE", max ? "max(data[tid],data[tid+stride])" : "data[tid]+data[tid+stride]")
        .replace("DEST", phase1 ? "wid.x" : "0u");
  }

  private static final String MATMUL =
      """
      struct Params { m:u32,k:u32,n:u32, }
      @group(0) @binding(0) var<storage,read_write> output: array<f32>;
      @group(0) @binding(1) var<storage,read> a: array<f32>;
      @group(0) @binding(2) var<storage,read> b: array<f32>;
      @group(0) @binding(3) var<storage,read> p: Params;
      var<workgroup> aa: array<f32,1024>;
      var<workgroup> bb: array<f32,1024>;
      @compute @workgroup_size(16,16)
      fn main(@builtin(local_invocation_id) lid: vec3<u32>,@builtin(workgroup_id) wid: vec3<u32>) {
        let tid=lid.y*16u+lid.x;
        let rowBase=wid.y*64u;let colBase=wid.x*64u;
        let rowTile=rowBase+lid.y*4u;let colTile=colBase+lid.x*4u;
        var acc:array<array<f32,4>,4>;
        for(var base=0u;base<p.k;base+=16u) {
          for(var i=0u;i<4u;i++) {
            let index=tid+i*256u;let sk=index&15u;let sm=index>>4u;
            let row=rowBase+sm;let k=base+sk;
            aa[sk*64u+sm]=0.;if(row<p.m&&k<p.k){aa[sk*64u+sm]=a[row*p.k+k];}
          }
          for(var i=0u;i<4u;i++) {
            let index=tid+i*256u;let sn=index&63u;let sk=index>>6u;
            let k=base+sk;let column=colBase+sn;
            bb[sk*64u+sn]=0.;if(k<p.k&&column<p.n){bb[sk*64u+sn]=b[k*p.n+column];}
          }
          workgroupBarrier();
          for(var k=0u;k<16u;k++) {
            var av:array<f32,4>;var bv:array<f32,4>;
            for(var i=0u;i<4u;i++){av[i]=aa[k*64u+lid.y*4u+i];}
            for(var j=0u;j<4u;j++){bv[j]=bb[k*64u+lid.x*4u+j];}
            for(var i=0u;i<4u;i++){for(var j=0u;j<4u;j++){acc[i][j]+=av[i]*bv[j];}}
          }
          workgroupBarrier();
        }
        for(var i=0u;i<4u;i++) {
          let row=rowTile+i;
          if(row<p.m){for(var j=0u;j<4u;j++){let column=colTile+j;if(column<p.n){output[row*p.n+column]=acc[i][j];}}}
        }
      }
      """;
  private static final String BATCHED_MATMUL =
      """
      struct Params { m:u32,k:u32,n:u32,batchA:u32,batchB:u32, }
      @group(0) @binding(0) var<storage,read_write> output: array<f32>;
      @group(0) @binding(1) var<storage,read> a: array<f32>;
      @group(0) @binding(2) var<storage,read> b: array<f32>;
      @group(0) @binding(3) var<storage,read> p: Params;
      var<workgroup> aa:array<f32,256>;var<workgroup> bb:array<f32,256>;
      @compute @workgroup_size(16,16)
      fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(global_invocation_id) gid:vec3<u32>) {
        let row=gid.y;let column=gid.x;let batch=gid.z;let tx=lid.x;let ty=lid.y;
        let aBase=select(batch,0u,p.batchA==1u)*p.m*p.k;let bBase=select(batch,0u,p.batchB==1u)*p.k*p.n;
        var acc=0.;
        for(var base=0u;base<p.k;base+=16u) {
          aa[ty*16u+tx]=0.;bb[ty*16u+tx]=0.;
          if(row<p.m&&base+tx<p.k){aa[ty*16u+tx]=a[aBase+row*p.k+base+tx];}
          if(column<p.n&&base+ty<p.k){bb[ty*16u+tx]=b[bBase+(base+ty)*p.n+column];}
          workgroupBarrier();
          for(var j=0u;j<16u;j++){acc+=aa[ty*16u+j]*bb[j*16u+tx];}
          workgroupBarrier();
        }
        if(row<p.m&&column<p.n){output[(batch*p.m+row)*p.n+column]=acc;}
      }
      """;
}
