; ModuleID = 'builtin.module'
source_filename = "fgn_oxide_kernels"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @bit_reverse_f32(ptr %v0, i64 %v1, i64 %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb5
bb1:
  %v10 = udiv i64 %v32, %v7
  %v11 = urem i64 %v32, %v7
  br label %bb8
bb2:
  %v12 = mul i64 %v10, %v7
  %v13 = add i64 %v12, %v11
  %v14 = mul i64 2, %v13
  %v15 = add i64 %v12, %v35
  %v16 = mul i64 2, %v15
  %v17 = extractvalue { ptr, i64 } %v6, 0
  %v18 = getelementptr inbounds float, ptr %v17, i64 %v14
  %v19 = load float, ptr %v18
  %v20 = add i64 %v14, 1
  %v21 = getelementptr inbounds float, ptr %v17, i64 %v20
  %v22 = load float, ptr %v21
  %v23 = getelementptr inbounds float, ptr %v17, i64 %v16
  %v24 = load float, ptr %v23
  store float %v24, ptr %v18
  %v25 = add i64 %v16, 1
  %v26 = getelementptr inbounds float, ptr %v17, i64 %v25
  %v27 = load float, ptr %v26
  store float %v27, ptr %v21
  store float %v19, ptr %v23
  store float %v22, ptr %v26
  br label %bb4
bb3:
  br label %bb4
bb4:
  ret void
bb5:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb6
bb6:
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb7
bb7:
  %v30 = mul i32 %v28, %v29
  %v31 = add i32 %v30, %v9
  %v32 = zext i32 %v31 to i64
  %v33 = icmp eq i64 %v7, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb1, label %bb11
bb8:
  %v35 = phi i64 [ 0, %bb1 ], [ %v44, %bb9 ]
  %v36 = phi i64 [ %v11, %bb1 ], [ %v47, %bb9 ]
  %v37 = phi i64 [ 0, %bb1 ], [ %v48, %bb9 ]
  %v38 = icmp ult i64 %v37, %v8
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb10, label %bb9
bb9:
  %v40 = zext i32 1 to i64
  %v41 = and i64 %v40, 63
  %v42 = shl i64 %v35, %v41
  %v43 = and i64 %v36, 1
  %v44 = or i64 %v42, %v43
  %v45 = zext i32 1 to i64
  %v46 = and i64 %v45, 63
  %v47 = lshr i64 %v36, %v46
  %v48 = add i64 %v37, 1
  br label %bb8
bb10:
  %v49 = icmp ult i64 %v11, %v35
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb3, label %bb2
bb11:
  unreachable
}

declare double @__nv_log(double)
declare double @__nv_cos(double)
declare double @__nv_sqrt(double)
declare double @__nv_sin(double)

define void @gen_scale_f64(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6, i64 %v7) {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi { ptr, i64 } [ %v11, %entry ]
  %v14 = phi i64 [ %v4, %entry ]
  %v15 = phi i64 [ %v5, %entry ]
  %v16 = phi i64 [ %v6, %entry ]
  %v17 = phi i64 [ %v7, %entry ]
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb1:
  %v19 = call { i32, i32 } @fgn_oxide_kernels__philox2x32_10(i64 %v42, i64 %v16, i64 %v17)
  br label %bb2
bb2:
  %v20 = extractvalue { i32, i32 } %v19, 0
  %v21 = extractvalue { i32, i32 } %v19, 1
  %v22 = uitofp i32 %v20 to double
  %v23 = fadd double %v22, 0.5
  %v24 = fmul double %v23, 0.00000000023283064365386963
  %v25 = uitofp i32 %v21 to double
  %v26 = fadd double %v25, 0.5
  %v27 = fmul double %v26, 0.00000000023283064365386963
  %v28 = call double @__nv_log(double %v24)
  br label %bb9
bb3:
  %v29 = urem i64 %v42, %v14
  %v30 = extractvalue { ptr, i64 } %v13, 1
  %v31 = icmp ult i64 %v29, %v30
  br i1 %v31, label %bb4, label %bb13
bb4:
  %v32 = extractvalue { ptr, i64 } %v13, 0
  %v33 = getelementptr inbounds double, ptr %v32, i64 %v29
  %v34 = load double, ptr %v33
  %v35 = mul i64 2, %v42
  %v36 = extractvalue { ptr, i64 } %v12, 0
  %v37 = call double @__nv_cos(double %v47)
  br label %bb11
bb5:
  ret void
bb6:
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb7
bb7:
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb8
bb8:
  %v40 = mul i32 %v38, %v39
  %v41 = add i32 %v40, %v18
  %v42 = zext i32 %v41 to i64
  %v43 = icmp ult i64 %v42, %v15
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb5, label %bb1
bb9:
  %v45 = fmul double -2.0, %v28
  %v46 = call double @__nv_sqrt(double %v45)
  br label %bb10
bb10:
  %v47 = fmul double 6.283185307179586, %v27
  %v48 = icmp eq i64 %v14, 0
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb3, label %bb14
bb11:
  %v50 = fmul double %v46, %v37
  %v51 = getelementptr inbounds double, ptr %v36, i64 %v35
  %v52 = fmul double %v50, %v34
  store double %v52, ptr %v51
  %v53 = call double @__nv_sin(double %v47)
  br label %bb12
bb12:
  %v54 = fmul double %v46, %v53
  %v55 = add i64 %v35, 1
  %v56 = getelementptr inbounds double, ptr %v36, i64 %v55
  %v57 = fmul double %v54, %v34
  store double %v57, ptr %v56
  br label %bb5
bb13:
  unreachable
bb14:
  unreachable
}

define void @bit_reverse_f64(ptr %v0, i64 %v1, i64 %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb5
bb1:
  %v10 = udiv i64 %v32, %v7
  %v11 = urem i64 %v32, %v7
  br label %bb8
bb2:
  %v12 = mul i64 %v10, %v7
  %v13 = add i64 %v12, %v11
  %v14 = mul i64 2, %v13
  %v15 = add i64 %v12, %v35
  %v16 = mul i64 2, %v15
  %v17 = extractvalue { ptr, i64 } %v6, 0
  %v18 = getelementptr inbounds double, ptr %v17, i64 %v14
  %v19 = load double, ptr %v18
  %v20 = add i64 %v14, 1
  %v21 = getelementptr inbounds double, ptr %v17, i64 %v20
  %v22 = load double, ptr %v21
  %v23 = getelementptr inbounds double, ptr %v17, i64 %v16
  %v24 = load double, ptr %v23
  store double %v24, ptr %v18
  %v25 = add i64 %v16, 1
  %v26 = getelementptr inbounds double, ptr %v17, i64 %v25
  %v27 = load double, ptr %v26
  store double %v27, ptr %v21
  store double %v19, ptr %v23
  store double %v22, ptr %v26
  br label %bb4
bb3:
  br label %bb4
bb4:
  ret void
bb5:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb6
bb6:
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb7
bb7:
  %v30 = mul i32 %v28, %v29
  %v31 = add i32 %v30, %v9
  %v32 = zext i32 %v31 to i64
  %v33 = icmp eq i64 %v7, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb1, label %bb11
bb8:
  %v35 = phi i64 [ 0, %bb1 ], [ %v44, %bb9 ]
  %v36 = phi i64 [ %v11, %bb1 ], [ %v47, %bb9 ]
  %v37 = phi i64 [ 0, %bb1 ], [ %v48, %bb9 ]
  %v38 = icmp ult i64 %v37, %v8
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb10, label %bb9
bb9:
  %v40 = zext i32 1 to i64
  %v41 = and i64 %v40, 63
  %v42 = shl i64 %v35, %v41
  %v43 = and i64 %v36, 1
  %v44 = or i64 %v42, %v43
  %v45 = zext i32 1 to i64
  %v46 = and i64 %v45, 63
  %v47 = lshr i64 %v36, %v46
  %v48 = add i64 %v37, 1
  br label %bb8
bb10:
  %v49 = icmp ult i64 %v11, %v35
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb3, label %bb2
bb11:
  unreachable
}

define void @extract_real_f64(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i64 %v4, i64 %v5, double %v6) {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi { ptr, i64 } [ %v10, %entry ]
  %v13 = phi i64 [ %v4, %entry ]
  %v14 = phi i64 [ %v5, %entry ]
  %v15 = phi double [ %v6, %entry ]
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb1:
  %v17 = extractvalue { i8, ptr } %v45, 1
  %v18 = icmp eq i64 %v13, 0
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb2, label %bb14
bb2:
  %v20 = udiv i64 %v36, %v13
  %v21 = urem i64 %v36, %v13
  %v22 = mul i64 %v20, %v14
  %v23 = add i64 %v22, %v21
  %v24 = add i64 %v23, 1
  %v25 = mul i64 2, %v24
  %v26 = extractvalue { ptr, i64 } %v11, 1
  %v27 = icmp ult i64 %v25, %v26
  br i1 %v27, label %bb3, label %bb15
bb3:
  %v28 = extractvalue { ptr, i64 } %v11, 0
  %v29 = getelementptr inbounds double, ptr %v28, i64 %v25
  %v30 = load double, ptr %v29
  %v31 = fmul double %v30, %v15
  store double %v31, ptr %v17
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v32 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb7
bb7:
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb8
bb8:
  %v34 = mul i32 %v32, %v33
  %v35 = add i32 %v34, %v16
  %v36 = zext i32 %v35 to i64
  %v37 = extractvalue { ptr, i64 } %v12, 1
  %v38 = icmp ult i64 %v36, %v37
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb10, label %bb9
bb9:
  %v40 = extractvalue { ptr, i64 } %v12, 0
  %v41 = getelementptr inbounds double, ptr %v40, i64 %v36
  %v42 = insertvalue { i8, ptr } undef, i8 1, 0
  %v43 = insertvalue { i8, ptr } %v42, ptr %v41, 1
  br label %bb11
bb10:
  %v44 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb11
bb11:
  %v45 = phi { i8, ptr } [ %v43, %bb9 ], [ %v44, %bb10 ]
  %v46 = extractvalue { i8, ptr } %v45, 0
  %v47 = zext i8 %v46 to i64
  %v48 = icmp eq i64 %v47, 1
  br i1 %v48, label %bb1, label %bb12
bb12:
  %v49 = icmp eq i64 %v47, 0
  br i1 %v49, label %bb4, label %bb13
bb13:
  unreachable
bb14:
  unreachable
bb15:
  unreachable
}

define void @extract_real_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i64 %v4, i64 %v5, float %v6) {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  br label %bb0
bb0:
  %v11 = phi { ptr, i64 } [ %v8, %entry ]
  %v12 = phi { ptr, i64 } [ %v10, %entry ]
  %v13 = phi i64 [ %v4, %entry ]
  %v14 = phi i64 [ %v5, %entry ]
  %v15 = phi float [ %v6, %entry ]
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb1:
  %v17 = extractvalue { i8, ptr } %v45, 1
  %v18 = icmp eq i64 %v13, 0
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb2, label %bb14
bb2:
  %v20 = udiv i64 %v36, %v13
  %v21 = urem i64 %v36, %v13
  %v22 = mul i64 %v20, %v14
  %v23 = add i64 %v22, %v21
  %v24 = add i64 %v23, 1
  %v25 = mul i64 2, %v24
  %v26 = extractvalue { ptr, i64 } %v11, 1
  %v27 = icmp ult i64 %v25, %v26
  br i1 %v27, label %bb3, label %bb15
bb3:
  %v28 = extractvalue { ptr, i64 } %v11, 0
  %v29 = getelementptr inbounds float, ptr %v28, i64 %v25
  %v30 = load float, ptr %v29
  %v31 = fmul float %v30, %v15
  store float %v31, ptr %v17
  br label %bb5
bb4:
  br label %bb5
bb5:
  ret void
bb6:
  %v32 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb7
bb7:
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb8
bb8:
  %v34 = mul i32 %v32, %v33
  %v35 = add i32 %v34, %v16
  %v36 = zext i32 %v35 to i64
  %v37 = extractvalue { ptr, i64 } %v12, 1
  %v38 = icmp ult i64 %v36, %v37
  %v39 = xor i1 %v38, 1
  br i1 %v39, label %bb10, label %bb9
bb9:
  %v40 = extractvalue { ptr, i64 } %v12, 0
  %v41 = getelementptr inbounds float, ptr %v40, i64 %v36
  %v42 = insertvalue { i8, ptr } undef, i8 1, 0
  %v43 = insertvalue { i8, ptr } %v42, ptr %v41, 1
  br label %bb11
bb10:
  %v44 = insertvalue { i8, ptr } undef, i8 0, 0
  br label %bb11
bb11:
  %v45 = phi { i8, ptr } [ %v43, %bb9 ], [ %v44, %bb10 ]
  %v46 = extractvalue { i8, ptr } %v45, 0
  %v47 = zext i8 %v46 to i64
  %v48 = icmp eq i64 %v47, 1
  br i1 %v48, label %bb1, label %bb12
bb12:
  %v49 = icmp eq i64 %v47, 0
  br i1 %v49, label %bb4, label %bb13
bb13:
  unreachable
bb14:
  unreachable
bb15:
  unreachable
}

declare float @__nv_logf(float)
declare float @__nv_cosf(float)
declare float @__nv_sqrtf(float)
declare float @__nv_sinf(float)

define void @gen_scale_f32(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i64 %v4, i64 %v5, i64 %v6, i64 %v7) {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  br label %bb0
bb0:
  %v12 = phi { ptr, i64 } [ %v9, %entry ]
  %v13 = phi { ptr, i64 } [ %v11, %entry ]
  %v14 = phi i64 [ %v4, %entry ]
  %v15 = phi i64 [ %v5, %entry ]
  %v16 = phi i64 [ %v6, %entry ]
  %v17 = phi i64 [ %v7, %entry ]
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb6
bb1:
  %v19 = call { i32, i32 } @fgn_oxide_kernels__philox2x32_10(i64 %v42, i64 %v16, i64 %v17)
  br label %bb2
bb2:
  %v20 = extractvalue { i32, i32 } %v19, 0
  %v21 = extractvalue { i32, i32 } %v19, 1
  %v22 = uitofp i32 %v20 to float
  %v23 = fadd float %v22, 0.5
  %v24 = fmul float %v23, 0.00000000023283064365386963
  %v25 = uitofp i32 %v21 to float
  %v26 = fadd float %v25, 0.5
  %v27 = fmul float %v26, 0.00000000023283064365386963
  %v28 = call float @__nv_logf(float %v24)
  br label %bb9
bb3:
  %v29 = urem i64 %v42, %v14
  %v30 = extractvalue { ptr, i64 } %v13, 1
  %v31 = icmp ult i64 %v29, %v30
  br i1 %v31, label %bb4, label %bb13
bb4:
  %v32 = extractvalue { ptr, i64 } %v13, 0
  %v33 = getelementptr inbounds float, ptr %v32, i64 %v29
  %v34 = load float, ptr %v33
  %v35 = mul i64 2, %v42
  %v36 = extractvalue { ptr, i64 } %v12, 0
  %v37 = call float @__nv_cosf(float %v47)
  br label %bb11
bb5:
  ret void
bb6:
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb7
bb7:
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb8
bb8:
  %v40 = mul i32 %v38, %v39
  %v41 = add i32 %v40, %v18
  %v42 = zext i32 %v41 to i64
  %v43 = icmp ult i64 %v42, %v15
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb5, label %bb1
bb9:
  %v45 = fmul float -2.0, %v28
  %v46 = call float @__nv_sqrtf(float %v45)
  br label %bb10
bb10:
  %v47 = fmul float 6.2831854820251465, %v27
  %v48 = icmp eq i64 %v14, 0
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb3, label %bb14
bb11:
  %v50 = fmul float %v46, %v37
  %v51 = getelementptr inbounds float, ptr %v36, i64 %v35
  %v52 = fmul float %v50, %v34
  store float %v52, ptr %v51
  %v53 = call float @__nv_sinf(float %v47)
  br label %bb12
bb12:
  %v54 = fmul float %v46, %v53
  %v55 = add i64 %v35, 1
  %v56 = getelementptr inbounds float, ptr %v36, i64 %v55
  %v57 = fmul float %v54, %v34
  store float %v57, ptr %v56
  br label %bb5
bb13:
  unreachable
bb14:
  unreachable
}

define void @fft_stage_f64(ptr %v0, i64 %v1, i64 %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb3
bb1:
  %v10 = udiv i64 %v31, %v32
  %v11 = urem i64 %v31, %v32
  %v12 = mul i64 %v8, 2
  %v13 = icmp eq i64 %v8, 0
  %v14 = xor i1 %v13, 1
  br i1 %v14, label %bb2, label %bb8
bb2:
  %v15 = udiv i64 %v11, %v8
  %v16 = urem i64 %v11, %v8
  %v17 = mul i64 %v10, %v7
  %v18 = mul i64 %v15, %v12
  %v19 = add i64 %v17, %v18
  %v20 = add i64 %v19, %v16
  %v21 = add i64 %v20, %v8
  %v22 = uitofp i64 %v16 to double
  %v23 = fmul double -6.283185307179586, %v22
  %v24 = uitofp i64 %v12 to double
  %v25 = fdiv double %v23, %v24
  %v26 = call double @__nv_cos(double %v25)
  br label %bb6
bb3:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb4
bb4:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb5
bb5:
  %v29 = mul i32 %v27, %v28
  %v30 = add i32 %v29, %v9
  %v31 = zext i32 %v30 to i64
  %v32 = udiv i64 %v7, 2
  %v33 = icmp eq i64 %v32, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb1, label %bb9
bb6:
  %v35 = call double @__nv_sin(double %v25)
  br label %bb7
bb7:
  %v36 = extractvalue { ptr, i64 } %v6, 0
  %v37 = mul i64 2, %v20
  %v38 = mul i64 2, %v21
  %v39 = getelementptr inbounds double, ptr %v36, i64 %v37
  %v40 = load double, ptr %v39
  %v41 = add i64 %v37, 1
  %v42 = getelementptr inbounds double, ptr %v36, i64 %v41
  %v43 = load double, ptr %v42
  %v44 = getelementptr inbounds double, ptr %v36, i64 %v38
  %v45 = load double, ptr %v44
  %v46 = add i64 %v38, 1
  %v47 = getelementptr inbounds double, ptr %v36, i64 %v46
  %v48 = load double, ptr %v47
  %v49 = fmul double %v45, %v26
  %v50 = fmul double %v48, %v35
  %v51 = fsub double %v49, %v50
  %v52 = fmul double %v45, %v35
  %v53 = fmul double %v48, %v26
  %v54 = fadd double %v52, %v53
  %v55 = fadd double %v40, %v51
  store double %v55, ptr %v39
  %v56 = fadd double %v43, %v54
  store double %v56, ptr %v42
  %v57 = fsub double %v40, %v51
  store double %v57, ptr %v44
  %v58 = fsub double %v43, %v54
  store double %v58, ptr %v47
  ret void
bb8:
  unreachable
bb9:
  unreachable
}

define void @fft_stage_f32(ptr %v0, i64 %v1, i64 %v2, i64 %v3) {
entry:
  %v4 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v5 = insertvalue { ptr, i64 } %v4, i64 %v1, 1
  br label %bb0
bb0:
  %v6 = phi { ptr, i64 } [ %v5, %entry ]
  %v7 = phi i64 [ %v2, %entry ]
  %v8 = phi i64 [ %v3, %entry ]
  %v9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb3
bb1:
  %v10 = udiv i64 %v31, %v32
  %v11 = urem i64 %v31, %v32
  %v12 = mul i64 %v8, 2
  %v13 = icmp eq i64 %v8, 0
  %v14 = xor i1 %v13, 1
  br i1 %v14, label %bb2, label %bb8
bb2:
  %v15 = udiv i64 %v11, %v8
  %v16 = urem i64 %v11, %v8
  %v17 = mul i64 %v10, %v7
  %v18 = mul i64 %v15, %v12
  %v19 = add i64 %v17, %v18
  %v20 = add i64 %v19, %v16
  %v21 = add i64 %v20, %v8
  %v22 = uitofp i64 %v16 to float
  %v23 = fmul float -6.2831854820251465, %v22
  %v24 = uitofp i64 %v12 to float
  %v25 = fdiv float %v23, %v24
  %v26 = call float @__nv_cosf(float %v25)
  br label %bb6
bb3:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb4
bb4:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb5
bb5:
  %v29 = mul i32 %v27, %v28
  %v30 = add i32 %v29, %v9
  %v31 = zext i32 %v30 to i64
  %v32 = udiv i64 %v7, 2
  %v33 = icmp eq i64 %v32, 0
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb1, label %bb9
bb6:
  %v35 = call float @__nv_sinf(float %v25)
  br label %bb7
bb7:
  %v36 = extractvalue { ptr, i64 } %v6, 0
  %v37 = mul i64 2, %v20
  %v38 = mul i64 2, %v21
  %v39 = getelementptr inbounds float, ptr %v36, i64 %v37
  %v40 = load float, ptr %v39
  %v41 = add i64 %v37, 1
  %v42 = getelementptr inbounds float, ptr %v36, i64 %v41
  %v43 = load float, ptr %v42
  %v44 = getelementptr inbounds float, ptr %v36, i64 %v38
  %v45 = load float, ptr %v44
  %v46 = add i64 %v38, 1
  %v47 = getelementptr inbounds float, ptr %v36, i64 %v46
  %v48 = load float, ptr %v47
  %v49 = fmul float %v45, %v26
  %v50 = fmul float %v48, %v35
  %v51 = fsub float %v49, %v50
  %v52 = fmul float %v45, %v35
  %v53 = fmul float %v48, %v26
  %v54 = fadd float %v52, %v53
  %v55 = fadd float %v40, %v51
  store float %v55, ptr %v39
  %v56 = fadd float %v43, %v54
  store float %v56, ptr %v42
  %v57 = fsub float %v40, %v51
  store float %v57, ptr %v44
  %v58 = fsub float %v43, %v54
  store float %v58, ptr %v47
  ret void
bb8:
  unreachable
bb9:
  unreachable
}

define { i32, i32 } @fgn_oxide_kernels__philox2x32_10(i64 %v0, i64 %v1, i64 %v2) {
entry:
  br label %bb0
bb0:
  %v3 = phi i64 [ %v0, %entry ]
  %v4 = phi i64 [ %v1, %entry ]
  %v5 = phi i64 [ %v2, %entry ]
  %v6 = bitcast i64 %v3 to i64
  %v7 = add i64 %v6, %v5
  %v8 = trunc i64 %v7 to i32
  %v9 = zext i32 32 to i64
  %v10 = and i64 %v9, 63
  %v11 = lshr i64 %v7, %v10
  %v12 = trunc i64 %v11 to i32
  %v13 = trunc i64 %v4 to i32
  br label %bb1
bb1:
  %v14 = phi i32 [ %v8, %bb0 ], [ %v27, %bb2 ]
  %v15 = phi i32 [ %v12, %bb0 ], [ %v28, %bb2 ]
  %v16 = phi i32 [ %v13, %bb0 ], [ %v29, %bb2 ]
  %v17 = phi i32 [ 0, %bb0 ], [ %v30, %bb2 ]
  %v18 = icmp slt i32 %v17, 10
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb3, label %bb2
bb2:
  %v20 = zext i32 %v14 to i64
  %v21 = mul i64 3528531795, %v20
  %v22 = zext i32 32 to i64
  %v23 = and i64 %v22, 63
  %v24 = lshr i64 %v21, %v23
  %v25 = trunc i64 %v24 to i32
  %v26 = xor i32 %v25, %v15
  %v27 = xor i32 %v26, %v16
  %v28 = trunc i64 %v21 to i32
  %v29 = add i32 %v16, 2654435769
  %v30 = add i32 %v17, 1
  br label %bb1
bb3:
  %v31 = insertvalue { i32, i32 } undef, i32 %v14, 0
  %v32 = insertvalue { i32, i32 } %v31, i32 %v15, 1
  ret { i32, i32 } %v32
}


@llvm.used = appending global [8 x ptr] [ptr @bit_reverse_f32, ptr @gen_scale_f64, ptr @bit_reverse_f64, ptr @extract_real_f64, ptr @extract_real_f32, ptr @gen_scale_f32, ptr @fft_stage_f64, ptr @fft_stage_f32], section "llvm.metadata"

!0 = !{ptr @bit_reverse_f32, !"kernel", i32 1}
!1 = !{ptr @gen_scale_f64, !"kernel", i32 1}
!2 = !{ptr @bit_reverse_f64, !"kernel", i32 1}
!3 = !{ptr @extract_real_f64, !"kernel", i32 1}
!4 = !{ptr @extract_real_f32, !"kernel", i32 1}
!5 = !{ptr @gen_scale_f32, !"kernel", i32 1}
!6 = !{ptr @fft_stage_f64, !"kernel", i32 1}
!7 = !{ptr @fft_stage_f32, !"kernel", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7}

!nvvmir.version = !{!8}
!8 = !{i32 2, i32 0, i32 3, i32 2}
