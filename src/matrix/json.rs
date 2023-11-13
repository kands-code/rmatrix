use std::io::Write;

use crate::{matrix::Matrix, types::Number};

impl<N: Number> Matrix<N> {
    pub fn from_json<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        Ok(serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path)?,
        ))?)
    }

    pub fn to_json<P: AsRef<std::path::Path>>(
        data: &Vec<Self>,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        write!(
            std::fs::File::create(path)?,
            "{}",
            serde_json::to_string(data)?
        )?;
        Ok(())
    }

    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        write!(
            std::fs::File::create(path)?,
            "{}",
            serde_json::to_string(&vec![self])?
        )?;
        Ok(())
    }
}
