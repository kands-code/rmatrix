#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{
        matrix::Matrix,
        utils::{horizontal_concat, vertical_concat},
    },
    number::instances::int::Int,
};

fn main() {
    let m1: Matrix<Int, 3, 3> = Matrix::rand(Int::of(-10), Int::of(10));
    let p = Matrix::<Int, 3, 3>::p_add(2, 1, Int::of(-1));
    let smal = p * m1.clone();
    let vcat = vertical_concat(&m1, &smal);
    let hcat = horizontal_concat(&m1, &smal);
    println!("{}", m1);
    println!("{}", smal);
    println!("{}", vcat);
    println!("{}", hcat);
}
