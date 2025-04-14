// //! # examples::main
// //!
// //! A simple demonstration of using this library.

// #![warn(missing_docs)]

// use rmatrix_ks::{
//     matrix::{
//         complex::{is_hermitian_matrix, moore_penrose_inverse},
//         matrix::Matrix,
//     },
//     number::instances::{complex::Complex, float::Float},
// };

// fn main() {
//     // Properties that the Moore-Penrose inverse must satisfy.
//     let rect = Matrix::<Complex<Float>>::of(
//         2,
//         3,
//         &[
//             (1.0, 2.0),
//             (2.0, -1.0),
//             (3.0, 0.0),
//             (4.0, 0.0),
//             (5.0, 1.0),
//             (6.0, -2.0),
//         ]
//         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
//     )
//     .unwrap();
//     let rectp = moore_penrose_inverse(&rect);
//     // A . Ap . A = A
//     assert_eq!(rect.clone() * rectp.clone() * rect.clone(), rect);
//     // Ap . A . Ap = Ap
//     assert_eq!(rectp.clone() * rect.clone() * rectp.clone(), rectp);
//     // A . Ap is a hermitian matrix.
//     assert!(is_hermitian_matrix(&(rect.clone() * rectp.clone())));
//     // Ap . A is also a hermitian matrix.
//     assert!(is_hermitian_matrix(&(rectp * rect)));
// }

use rmatrix_ks::number::instances::double::Double;

fn main() {
    let d = Double::of(3.0);
    assert!((d.raw() - 3.0).abs() < f64::EPSILON.sqrt());
}
