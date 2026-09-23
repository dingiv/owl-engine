//! 对比与契约检查。
//!
//! allclose 语义:`|got - want| <= atol + rtol * |want|`(numpy 约定);
//! NaN 恒不相等,除非 `nan_eq = true` 且双方同为 NaN。
//! DiffReport 报首个失配点(全量统计 + 现场),失败信息自带上下文。

use std::fmt;

/// 失配报告:首个坏点 + 全量统计。
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub op: String,
    pub first_index: Option<usize>,
    pub first_got: f32,
    pub first_want: f32,
    pub mismatch_count: usize,
    pub nan_got: usize,
    pub nan_want: usize,
    pub max_abs_err: f32,
    pub total: usize,
}

impl fmt::Display for DiffReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "op={} total={} 失配 {} 个(NaN: got {} / want {})| 首坏点 idx={} got={} want={} rel={:.3e} | max_abs_err={:.3e}",
            self.op,
            self.total,
            self.mismatch_count,
            self.nan_got,
            self.nan_want,
            self.first_index.map(|i| i.to_string()).unwrap_or("-".into()),
            self.first_got,
            self.first_want,
            if self.first_want != 0.0 {
                ((self.first_got - self.first_want) / self.first_want).abs()
            } else {
                f32::INFINITY
            },
            self.max_abs_err,
        )
    }
}

/// numpy 约定 allclose。`nan_eq = true` 时双方同 NaN 记相等。
pub fn allclose(
    op: &str,
    got: &[f32],
    want: &[f32],
    rtol: f32,
    atol: f32,
    nan_eq: bool,
) -> Result<(), DiffReport> {
    assert_eq!(
        got.len(),
        want.len(),
        "allclose({op}): 长度不一致 got {} vs want {}",
        got.len(),
        want.len()
    );
    let mut rep = DiffReport {
        op: op.to_string(),
        first_index: None,
        first_got: 0.0,
        first_want: 0.0,
        mismatch_count: 0,
        nan_got: 0,
        nan_want: 0,
        max_abs_err: 0.0,
        total: got.len(),
    };
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if g.is_nan() {
            rep.nan_got += 1;
        }
        if w.is_nan() {
            rep.nan_want += 1;
        }
        let both_nan = g.is_nan() && w.is_nan();
        let ok = if nan_eq && both_nan {
            true
        } else if g.is_nan() || w.is_nan() {
            false
        } else {
            (g - w).abs() <= atol + rtol * w.abs()
        };
        if !ok {
            rep.mismatch_count += 1;
            let err = if g.is_nan() || w.is_nan() {
                f32::INFINITY
            } else {
                (g - w).abs()
            };
            if err > rep.max_abs_err {
                rep.max_abs_err = err;
            }
            if rep.first_index.is_none() {
                rep.first_index = Some(i);
                rep.first_got = g;
                rep.first_want = w;
            }
        }
    }
    if rep.mismatch_count == 0 && rep.nan_got == rep.nan_want {
        Ok(())
    } else {
        Err(rep)
    }
}

/// 位型严格相等(NaN 位型也须一致;确定性判别用)。
pub fn bits_equal(op: &str, got: &[f32], want: &[f32]) -> Result<(), String> {
    assert_eq!(got.len(), want.len(), "bits_equal({op}): 长度不一致");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "bits_equal({op}): idx={i} got={g:?} want={w:?}(位型不一致)"
        );
    }
    Ok(())
}

/// 形状契约:失配信息携带算子名与全部输入形状
/// (narrow 丢 after / 模板序错位一类坑要在边界暴露)。
pub fn expect_shape(
    op: &str,
    got: &[usize],
    want: &[usize],
    input_shapes: &[(&str, &[usize])],
) -> Result<(), String> {
    if got == want {
        return Ok(());
    }
    let inputs = input_shapes
        .iter()
        .map(|(n, s)| format!("{n}:{s:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "形状契约失配 [{op}]: 输出 got {got:?} want {want:?}(输入 = {inputs})"
    ))
}

/// 有限性断言(首坏点 + 计数)。
pub fn assert_finite(op: &str, v: &[f32]) -> Result<(), String> {
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    if bad == 0 {
        return Ok(());
    }
    let first = v.iter().position(|x| !x.is_finite()).unwrap();
    Err(format!(
        "有限性失配 [{op}]: {bad}/{len} 非有限,首坏点 idx={first} 值={:?}",
        v[first],
        len = v.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allclose_pass_and_nan_policy() {
        assert!(allclose("t", &[1.0, 2.0], &[1.0 + 1e-7, 2.0], 1e-5, 1e-6, false).is_ok());
        // NaN 默认不相等(哪怕双方都 NaN)
        assert!(allclose("t", &[f32::NAN], &[f32::NAN], 1e-5, 1e-6, false).is_err());
        // nan_eq 开关放行双方同 NaN
        assert!(allclose("t", &[f32::NAN], &[f32::NAN], 1e-5, 1e-6, true).is_ok());
        // 单边 NaN 恒失配
        assert!(allclose("t", &[f32::NAN], &[1.0], 1e-5, 1e-6, true).is_err());
    }

    #[test]
    fn allclose_report_has_first_bad_point() {
        let got = [1.0, 2.0, 100.0, 4.0];
        let want = [1.0, 2.0, 3.0, 4.0];
        let err = allclose("demo", &got, &want, 1e-5, 1e-6, false).unwrap_err();
        assert_eq!(err.first_index, Some(2));
        assert_eq!(err.mismatch_count, 1);
        assert_eq!(err.first_got, 100.0);
        assert_eq!(err.first_want, 3.0);
    }

    #[test]
    fn expect_shape_carries_inputs() {
        let err = expect_shape(
            "narrow(dim=1)",
            &[2, 2, 4],
            &[2, 3, 4],
            &[("src", &[2, 3, 4]), ("weight", &[6144, 1, 4])],
        )
        .unwrap_err();
        assert!(err.contains("src:[2, 3, 4]"), "{err}");
        assert!(err.contains("narrow"), "{err}");
    }

    #[test]
    fn finite_catches_nan() {
        assert!(assert_finite("ok", &[1.0, 0.0, -3.5]).is_ok());
        let err = assert_finite("bad", &[1.0, f32::NAN]).unwrap_err();
        assert!(err.contains("1/2"), "{err}");
    }
}
