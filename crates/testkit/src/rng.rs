//! seed 确定性 RNG(splitmix64;零依赖,跨机跨进程同序列)。
//!
//! f32 生成:取 64 位输出的高 24 位 × 2^-24 归一到 [0,1),再仿射到区间。
//! normal 用 Box-Muller(两 uniform 出一对,缓存次序确定)。

/// splitmix64 状态机(seed 任意,含 0;黄金比例增量打散弱 seed)。
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// [0,1) 均匀 f32(24 位精度;不含 1.0)
    pub fn unit_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
    }

    /// [lo, hi) 均匀 f32
    pub fn f32_uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit_f32() * (hi - lo)
    }

    /// N(mu, sigma) 正态(Box-Muller;每次调用消耗 2 个 uniform,缓存另一半)
    pub fn f32_normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let u1 = self.unit_f32().max(1e-12); // 防 ln(0)
        let u2 = self.unit_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        mu + sigma * r * theta.sin() // 只取正弦支,余弦支弃用(次序仍确定)
    }

    /// [lo, hi) 均匀 u32
    pub fn u32_range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % ((hi - lo) as u64)) as u32
    }

    /// 填充一个切片(逐元素 uniform)
    pub fn fill_f32(&mut self, out: &mut [f32], lo: f32, hi: f32) {
        for v in out.iter_mut() {
            *v = self.f32_uniform(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.f32_uniform(-1.0, 1.0).to_bits(), b.f32_uniform(-1.0, 1.0).to_bits());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let va: Vec<f32> = (0..16).map(|_| a.unit_f32()).collect();
        let vb: Vec<f32> = (0..16).map(|_| b.unit_f32()).collect();
        assert_ne!(va, vb);
    }

    #[test]
    fn uniform_in_range_and_normal_finite() {
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let v = r.f32_uniform(-2.5, 3.5);
            assert!((-2.5..3.5).contains(&v));
            let n = r.f32_normal(0.0, 1.0);
            assert!(n.is_finite());
        }
    }
}
