use rmatrix_ks::number::{instances::integer::Integer, traits::zero::Zero};

fn main() {
    let a = Integer::zero();
    assert!(a.is_zero());
    let Some(b) = Integer::of(true, &[1, 6]) else {
        unreachable!();
    };
    assert!(!b.is_zero());
}
