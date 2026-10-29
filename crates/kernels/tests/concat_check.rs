use owl_kernels::family::marlin::repack::{
    marlin_gather_indices, pack_marlin_b_gather_into, pack_marlin_s,
};

fn pack_b(q: &[u8], k: usize, n: usize) -> Vec<i32> {
    let idx = marlin_gather_indices(k, n);
    let mut b = Vec::new();
    pack_marlin_b_gather_into(q, &idx, n * k / 8, &mut b);
    b
}

/// marlin B 的 N 轴有 tile 交错 ⇒ **半矩阵 B 不能拼接**。
/// W4A16Source 的合并键必须先拼原始 nibble 再整体重排；本门防“优化”回退。
#[test]
fn marlin_b_does_not_concatenate_across_n() {
    for &(k, half) in &[(128usize, 256usize), (2560usize, 9216usize)] {
        let qg: Vec<u8> = (0..k * half).map(|i| (i % 16) as u8).collect();
        let qu: Vec<u8> = (0..k * half).map(|i| ((i * 7 + 3) % 16) as u8).collect();
        let bg = pack_b(&qg, k, half);
        let bu = pack_b(&qu, k, half);
        let merged_q: Vec<u8> = qg.iter().chain(qu.iter()).copied().collect();
        let bm = pack_b(&merged_q, k, half * 2);
        let mut cat = bg;
        cat.extend(bu);
        assert_ne!(cat, bm, "k={k} n={half}: marlin B 竟然可以跨 N 拼接？");
    }
}

/// marlin S 的 64 元素 perm 块按**整行 out** 排布，out 变了边界就变 ⇒
/// 已重排的半尺度同样不能拼接；合并键必须先拼原始 [out, groups] 再整体重排。
#[test]
fn marlin_s_does_not_concatenate_across_n() {
    let (k, half) = (2560usize, 9216usize);
    let groups = k / 128;
    let sg: Vec<f32> = (0..half * groups).map(|i| (i as f32 + 1.0) * 1e-3).collect();
    let su: Vec<f32> = (0..half * groups).map(|i| (i as f32 + 2.0) * 1e-3).collect();
    let pg = pack_marlin_s(&sg, half, groups);
    let pu = pack_marlin_s(&su, half, groups);
    let mut merged_s = sg.clone();
    merged_s.extend(su.iter().copied());
    let pm = pack_marlin_s(&merged_s, half * 2, groups);
    let mut cat = pg;
    cat.extend(pu);
    assert_ne!(cat, pm, "marlin S 竟然可以跨 N 拼接？请复核 W4A16 合并键实现");
}
