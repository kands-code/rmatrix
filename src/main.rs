use rmatrix_ks::number::instances::integer::Integer;

fn main() {
    // 512
    let a = Integer::of(true, &[5, 1, 2]);
    // 512
    let another_a = Integer::of(true, &[5, 1, 2]);
    // -128
    let b = Integer::of(false, &[1, 2, 8]);
    // 1024
    let c = Integer::of(true, &[1, 0, 2, 4]);
    assert_eq!(a.partial_cmp(&another_a), Some(std::cmp::Ordering::Equal));
    assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Greater));
    assert_eq!(a.partial_cmp(&c), Some(std::cmp::Ordering::Less));
}
