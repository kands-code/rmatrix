use rmatrix_ks::{
    error::Result,
    matrix::Matrix,
    vector::{times_d, VectorC, VectorR},
};

#[cfg(feature = "rand_mat")]
const INTEGER_RANGE: std::ops::RangeInclusive<i32> = -128..=127;

#[test]
fn matrix_addition() -> Result<()> {
    let m1 = Matrix::<i32, 2, 3>::create(vec![2, 1, 0, 1, 3, 5])?;
    let m2 = Matrix::<i32, 2, 3>::create(vec![1, 0, 3, 4, -3, 1])?;
    assert_eq!(Matrix::create(vec![3, 1, 3, 5, 0, 6])?, m1.plus(m2)?);

    Ok(())
}

#[cfg(feature = "rand_mat")]
#[test]
/// A + B = B + A
fn matrix_addition_is_commutative() -> Result<()> {
    let m1 = Matrix::<i32, 2, 2>::rand(INTEGER_RANGE)?;
    let m2 = Matrix::<i32, 2, 2>::rand(INTEGER_RANGE)?;
    assert_eq!(m1.to_owned().plus(m2.to_owned())?, m2.plus(m1)?);

    Ok(())
}

#[test]
fn matrix_scalar_multiplication() -> Result<()> {
    let m1 = Matrix::<i32, 2, 3>::create(vec![2, 1, 0, 1, 3, 5])?;
    let m2 = Matrix::<i32, 2, 3>::create(vec![4, 2, 0, 2, 6, 10])?;
    assert_eq!(m2, m1.muls(2)?);

    Ok(())
}

#[test]
/// (A B) C = A (B C)
fn matrix_multiplication_is_associative() -> Result<()> {
    let a = VectorC::<i32, 2>::create(vec![1, 2])?;
    let b = VectorR::<i32, 3>::create(vec![1, 0, 1])?;
    let c = Matrix::<i32, 3, 2>::create(vec![2, 0, 1, 1, 0, 1])?;

    assert_eq!(
        a.to_owned().times(b.to_owned())?.times(c.to_owned())?,
        a.times(b.times(c)?)?
    );

    Ok(())
}

#[cfg(feature = "rand_mat")]
#[test]
/// (c A) B = A (c B)
fn matrix_multiplication_with_scalar_is_acceptable() -> Result<()> {
    use rand::Rng;

    let m1 = Matrix::<i32, 2, 2>::rand(INTEGER_RANGE)?;
    let m2 = Matrix::<i32, 2, 2>::rand(INTEGER_RANGE)?;
    let c = rand::thread_rng().gen_range(INTEGER_RANGE);

    assert_eq!(
        m1.to_owned().muls(c)?.times(m2.to_owned())?,
        m1.times(m2.muls(c)?)?
    );

    Ok(())
}

#[test]
fn row_vector_times_col_vector() -> Result<()> {
    assert_eq!(
        18,
        times_d(
            VectorR::<i32, 3>::create(vec![1, 3, 5])?,
            VectorC::create(vec![1, -1, 4])?
        )?
    );

    Ok(())
}
