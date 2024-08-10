use rmatrix_ks::column_vector::identity_vector_column;
use rmatrix_ks::column_vector::ColumnVector;
use rmatrix_ks::error::Result;

fn main() -> Result<()> {
    let x = identity_vector_column::<f32>(4, 2)?;
    assert_eq!(ColumnVector::create(4, vec![0.0, 1.0, 0.0, 0.0])?, x);
    Ok(())
}
