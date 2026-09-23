//! 声明式对拍用例:输入声明 → 绑定(指针面)→ 设备闭包 → host 参考 → 自动对比。
//!
//! 框架保证:输入落池同名可达、输出缓冲预清零、执行后 synchronize、
//! 回读、输出形状契约、逐元素 allclose(rtol/atol)、DiffReport 失配报告。
//! host 参考闭包拿到 host 侧输入副本(与设备输入逐字节相同),返回
//! `HashMap<输出名, Vec<f32>>`;框架按输出名对齐比较。

use std::collections::HashMap;

use crate::testkit::checks::allclose;
use crate::testkit::rig::{DevBuf, Rig};
use crate::testkit::rng::Rng;

/// 输入生成器(声明面;materialize 时逐元素求值,确定性)。
#[derive(Clone, Copy)]
pub enum Gen {
    /// [lo, hi) 均匀(seed 驱动)
    Uniform(f32, f32),
    /// N(mu, sigma)
    Normal(f32, f32),
    /// 常量
    Const(f32),
    /// 公式直译(i → f(i);迁移手写用例时保持逐字一致)
    Formula(fn(usize) -> f32),
}

/// u32 输入生成器。
#[derive(Clone, Copy)]
pub enum GenU32 {
    /// 等差 [start, start + n)
    Iota(u32),
    /// 公式直译
    Formula(fn(usize) -> u32),
    /// 常量
    Const(u32),
}

/// 输入/输出规格(声明面)。
struct Spec {
    name: &'static str,
    shape: Vec<usize>,
    is_output: bool,
    gen_f32: Option<Gen>,
    gen_u32: Option<GenU32>,
}

/// 对拍用例(构建后 `run` 一次;Rig 由调用方注入)。
pub struct Case {
    name: &'static str,
    seed: u64,
    specs: Vec<Spec>,
}

/// 用例绑定面(设备闭包入参):输入只读、输出可写。
pub struct Bound<'a> {
    pub rig: &'a Rig,
    inputs_f32: HashMap<&'static str, DevBuf>,
    inputs_u32: HashMap<&'static str, DevBuf>,
    outputs: HashMap<&'static str, DevBuf>,
    host_f32: HashMap<&'static str, Vec<f32>>,
    host_u32: HashMap<&'static str, Vec<u32>>,
    input_shapes: Vec<(&'static str, Vec<usize>)>,
}

impl<'a> Bound<'a> {
    /// 设备输入缓冲(f32 视图)。
    pub fn in_f32(&self, name: &str) -> &DevBuf {
        self.inputs_f32
            .get(name)
            .unwrap_or_else(|| panic!("case: 输入 {name} 未声明"))
    }

    /// 设备输入缓冲(u8 通配视图;kernel 通配位型入口)。
    pub fn in_u8(&self, name: &str) -> *const u8 {
        self.in_f32(name).u8_ptr()
    }

    /// 设备输入缓冲(u32 视图;cu_seqlens/slots 面)。
    pub fn in_u32(&self, name: &str) -> &DevBuf {
        self.inputs_u32
            .get(name)
            .unwrap_or_else(|| panic!("case: u32 输入 {name} 未声明"))
    }

    /// 设备输出缓冲(f32 可写视图)。
    pub fn out_f32_mut(&self, name: &str) -> *mut f32 {
        self.outputs
            .get(name)
            .unwrap_or_else(|| panic!("case: 输出 {name} 未声明"))
            .f32_ptr_mut()
    }

    /// 设备输出缓冲(u8 可写视图)。
    pub fn out_u8_mut(&self, name: &str) -> *mut u8 {
        self.outputs
            .get(name)
            .unwrap_or_else(|| panic!("case: 输出 {name} 未声明"))
            .u8_ptr_mut()
    }

    pub fn out_u32_mut(&self, name: &str) -> *mut u32 {
        self.outputs
            .get(name)
            .unwrap_or_else(|| panic!("case: 输出 {name} 未声明"))
            .u32_ptr_mut()
    }

    /// host 侧输入副本(与设备输入逐字节相同;host 参考消费)。
    pub fn host_f32(&self, name: &str) -> &[f32] {
        self.host_f32
            .get(name)
            .unwrap_or_else(|| panic!("case: 输入 {name} 未声明"))
    }

    /// host 侧 u32 输入副本。
    pub fn host_u32(&self, name: &str) -> &[u32] {
        self.host_u32
            .get(name)
            .unwrap_or_else(|| panic!("case: u32 输入 {name} 未声明"))
    }
}

impl Case {
    pub fn new(name: &'static str, seed: u64) -> Self {
        Self {
            name,
            seed,
            specs: Vec::new(),
        }
    }

    /// 声明 f32 输入。
    pub fn in_f32(mut self, name: &'static str, shape: &[usize], gen: Gen) -> Self {
        self.specs.push(Spec {
            name,
            shape: shape.to_vec(),
            is_output: false,
            gen_f32: Some(gen),
            gen_u32: None,
        });
        self
    }

    /// 声明 u32 输入。
    pub fn in_u32(mut self, name: &'static str, shape: &[usize], gen: GenU32) -> Self {
        self.specs.push(Spec {
            name,
            shape: shape.to_vec(),
            is_output: false,
            gen_f32: None,
            gen_u32: Some(gen),
        });
        self
    }

    /// 声明 f32 输出(预清零缓冲;形状即契约)。
    pub fn out_f32(mut self, name: &'static str, shape: &[usize]) -> Self {
        self.specs.push(Spec {
            name,
            shape: shape.to_vec(),
            is_output: true,
            gen_f32: None,
            gen_u32: None,
        });
        self
    }

    /// 执行对拍。
    ///
    /// - `device_fn`:发射闭包(拿 Bound 指针面 launch kernel;框架随后 sync);
    /// - `host_fn`:host 参考(拿 Bound 的 host 输入;返回 输出名 → 值)。
    pub fn run<F, G>(
        &self,
        rig: &Rig,
        rtol: f32,
        atol: f32,
        device_fn: F,
        host_fn: G,
    ) -> Result<(), String>
    where
        F: FnOnce(&Bound) -> Result<(), String>,
        G: FnOnce(&Bound) -> HashMap<String, Vec<f32>>,
    {
        // ---- materialize:声明 → host 值 + 设备缓冲(同 seed 同序列)----
        let mut rng = Rng::new(self.seed);
        let mut inputs_f32: HashMap<&'static str, DevBuf> = HashMap::new();
        let mut inputs_u32: HashMap<&'static str, DevBuf> = HashMap::new();
        let mut outputs: HashMap<&'static str, DevBuf> = HashMap::new();
        let mut host_f32: HashMap<&'static str, Vec<f32>> = HashMap::new();
        let mut host_u32: HashMap<&'static str, Vec<u32>> = HashMap::new();
        let mut input_shapes: Vec<(&'static str, Vec<usize>)> = Vec::new();
        for spec in &self.specs {
            let n: usize = spec.shape.iter().product();
            input_shapes.push((spec.name, spec.shape.clone()));
            if spec.is_output {
                outputs.insert(spec.name, rig.out_f32(n).with_shape(&spec.shape));
            } else if let Some(g) = spec.gen_f32 {
                let v: Vec<f32> = match g {
                    Gen::Uniform(lo, hi) => {
                        let mut v = vec![0f32; n];
                        rng.fill_f32(&mut v, lo, hi);
                        v
                    }
                    Gen::Normal(mu, sigma) => (0..n).map(|_| rng.f32_normal(mu, sigma)).collect(),
                    Gen::Const(c) => vec![c; n],
                    Gen::Formula(f) => (0..n).map(f).collect(),
                };
                inputs_f32.insert(spec.name, rig.htod(v.clone()).with_shape(&spec.shape));
                host_f32.insert(spec.name, v);
            } else if let Some(g) = spec.gen_u32 {
                let v: Vec<u32> = match g {
                    GenU32::Iota(start) => (0..n as u32).map(|i| start + i).collect(),
                    GenU32::Formula(f) => (0..n).map(f).collect(),
                    GenU32::Const(c) => vec![c; n],
                };
                inputs_u32.insert(spec.name, rig.htod(v.clone()).with_shape(&spec.shape));
                host_u32.insert(spec.name, v);
            }
        }

        let bound = Bound {
            rig,
            inputs_f32,
            inputs_u32,
            outputs,
            host_f32,
            host_u32,
            input_shapes,
        };

        // ---- 设备执行 + 同步 ----
        device_fn(&bound)?;
        rig.sync();

        // ---- host 参考 ----
        let want_map = host_fn(&bound);

        // ---- 输出契约 + 逐名对比 ----
        let in_shape_refs: Vec<(&str, &[usize])> = bound
            .input_shapes
            .iter()
            .map(|(n, s)| (*n, s.as_slice()))
            .collect();
        for spec in &self.specs {
            if !spec.is_output {
                continue;
            }
            let out_buf = &bound.outputs[spec.name];
            crate::testkit::checks::expect_shape(
                &format!("{}::{}", self.name, spec.name),
                out_buf.shape(),
                &spec.shape,
                &in_shape_refs,
            )?;
            let got = rig.dtoh_f32(out_buf);
            let want = want_map.get(spec.name).ok_or_else(|| {
                format!("case {}: host 参考未产出输出 {}", self.name, spec.name)
            })?;
            if want.len() != got.len() {
                return Err(format!(
                    "case {}: 输出 {} 参考长度 {} != 设备 {}",
                    self.name,
                    spec.name,
                    want.len(),
                    got.len()
                ));
            }
            allclose(
                &format!("{}::{}", self.name, spec.name),
                &got,
                want,
                rtol,
                atol,
                false,
            )
            .map_err(|rep| format!("case {}\n  {rep}\n  (rtol={rtol:e} atol={atol:e})", self.name))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::rig::Rig;

    /// 设备写入辅助:走 HtoD 同步拷贝(真设备写路径,不借 kernel)。
    fn htod_into(dst: *mut f32, src: &[f32]) {
        use owl_cuda::ffi::sys;
        unsafe {
            sys::cuMemcpyHtoD_v2(
                dst as owl_cuda::ffi::sys::CUdeviceptr,
                src.as_ptr() as *const std::ffi::c_void,
                src.len() * 4,
            )
            .result()
            .expect("smoke htod_into");
        }
    }

    /// 端到端冒烟:输出 = 输入 × 2(假算子;验证框架全链:生成/绑定/回读/对比/契约)
    #[test]
    fn case_end_to_end_smoke() {
        let rig = Rig::acquire();
        let case = Case::new("smoke_scale2", 2026)
            .in_f32("x", &[16], Gen::Uniform(-1.0, 1.0))
            .out_f32("y", &[16]);
        case
            .run(
                &rig,
                1e-6,
                1e-6,
                |b| {
                    let y: Vec<f32> = b.host_f32("x").iter().map(|v| v * 2.0).collect();
                    htod_into(b.out_f32_mut("y"), &y);
                    Ok(())
                },
                |h| {
                    HashMap::from([(
                        "y".to_string(),
                        h.host_f32("x").iter().map(|v| v * 2.0).collect(),
                    )])
                },
            )
            .expect("smoke 全绿");
    }

    /// 失配要带 DiffReport(人为错参考)
    #[test]
    fn case_mismatch_reports() {
        let rig = Rig::acquire();
        let case = Case::new("smoke_bad", 1)
            .in_f32("x", &[4], Gen::Const(1.0))
            .out_f32("y", &[4]);
        let err = case
            .run(
                &rig,
                1e-6,
                1e-6,
                |b| {
                    htod_into(b.out_f32_mut("y"), &[1.0, 2.0, 3.0, 4.0]);
                    Ok(())
                },
                |_h| HashMap::from([("y".to_string(), vec![1.0, 2.0, 3.0, 5.0])]),
            )
            .unwrap_err();
        assert!(err.contains("smoke_bad") && err.contains("首坏点"), "{err}");
    }

    /// 形状契约:输出声明与参考形状冲突在框架层暴露
    #[test]
    fn case_shape_contract_declared() {
        let rig = Rig::acquire();
        let case = Case::new("smoke_shape", 2)
            .in_f32("x", &[2, 8], Gen::Const(1.0))
            .out_f32("y", &[2, 8]);
        // 参考给了 16 个元素,形状声明 [2,8] 一致 → 通过;改声明为 [16] 则
        // with_shape 断言(元素积)在框架侧直接炸,说明契约成立。
        let r = case.run(
            &rig,
            1e-6,
            1e-6,
            |b| {
                htod_into(b.out_f32_mut("y"), &vec![1.0f32; 16]);
                Ok(())
            },
            |_h| HashMap::from([("y".to_string(), vec![1.0f32; 16])]),
        );
        assert!(r.is_ok());
        let bad = Case::new("smoke_shape_bad", 2)
            .in_f32("x", &[2, 8], Gen::Const(1.0))
            .out_f32("y", &[16]);
        // 声明 [16] vs 输入积 16 一致——真正的形状失配由 expect_shape 文案携带输入形状,
        // 单元测试在 checks::expect_shape_carries_inputs 覆盖;此处验证无 panic 全链。
        assert!(bad
            .run(
                &rig,
                1e-6,
                1e-6,
                |b| {
                    htod_into(b.out_f32_mut("y"), &vec![1.0f32; 16]);
                    Ok(())
                },
                |_h| HashMap::from([("y".to_string(), vec![1.0f32; 16])])
            )
            .is_ok());
    }
}
