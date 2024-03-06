//! read and write file actions

use std::fs::File;
use std::path::Path;

use serde::{ser::SerializeStruct, Serialize};

use crate::number::INum;
use crate::Matrix;

pub trait IFile<'de> {
    /// operation value type
    type Val;

    /// store data to file
    fn write_to(path_to_file: &'de str, data: &'de Self::Val);

    /// read data from file
    fn read_from(file_content: &'de str) -> Self::Val;
}

impl<'de, N: INum<'de>, const R: usize, const C: usize> IFile<'de> for Matrix<N, R, C> {
    type Val = Vec<Self>;

    fn write_to(path_to_file: &'de str, data: &'de Self::Val) {
        let file = File::create(Path::new(path_to_file)).unwrap();
        serde_json::to_writer(file, &data).unwrap();
    }

    fn read_from(file_content: &'de str) -> Self::Val {
        serde_json::from_str(file_content).unwrap()
    }
}

impl<'de, N: INum<'de>, const R: usize, const C: usize> Serialize for Matrix<N, R, C> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_struct("Matrix", 2)?;
        s.serialize_field("data", &self.data)?;
        s.serialize_field("size", &(R, C))?;
        s.end()
    }
}
