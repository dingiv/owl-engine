#[test]
fn bench_probe() {
    // ① 4MB 热复写(对照之前的"7ns"假象)
    eprintln!("[bench-helper] 4MB 复写×300: {:.2}ns/元素", owl_cpu::bench_write(1_000_000, 300));
    // ② 268MB 首次触碰复写(装载块规模)
    eprintln!("[bench-helper] 268MB 复写×1: {:.2}ns/元素", owl_cpu::bench_write(67_000_000, 1));
    // ③ 134MB 首次读
    eprintln!("[bench-helper] 134MB 首次读: {:.2}ns/字节", owl_cpu::bench_read(134_000_000));
    // ④ 5MB 首次读(单次 upload 的 staging 规模)
    eprintln!("[bench-helper] 5MB 首次读: {:.2}ns/字节", owl_cpu::bench_read(5_000_000));
}
