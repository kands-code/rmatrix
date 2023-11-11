use crate::matrix::Matrix;
#[cfg(feature = "serialize")]
use std::io::Write;

impl Matrix {
    pub fn size(&self) -> (usize, usize) {
        (self.shape.row, self.shape.col)
    }

    pub fn get(&self, prow: usize, pcol: usize) -> Result<f64, String> {
        Ok(self.data[self.shape.vpos(prow, pcol)?].to_owned())
    }

    pub fn get_row(&self, r: usize) -> Result<Vec<f64>, String> {
        if r > self.shape.row || r == 0 {
            Err(format!("the row {} is out of boundary", r))
        } else {
            let mut mrow = Vec::new();
            for c in 1..=self.shape.col {
                mrow.push(self.get(r, c)?);
            }
            Ok(mrow)
        }
    }

    pub fn get_col(&self, c: usize) -> Result<Vec<f64>, String> {
        if c > self.shape.col || c == 0 {
            Err(format!("the column {} is out of boundary", c))
        } else {
            let mut mcol = Vec::new();
            for r in 1..=self.shape.row {
                mcol.push(self.get(r, c)?);
            }
            Ok(mcol)
        }
    }

    pub fn set(&mut self, elem: f64, prow: usize, pcol: usize) -> Result<(), String> {
        self.data[self.shape.vpos(prow, pcol)?] = elem;
        Ok(())
    }

    pub fn trace(&self) -> Result<f64, String> {
        let mut t = 0.0;
        for i in 1..=self.shape.row.min(self.shape.col) {
            t += self.get(i, i)?;
        }
        Ok(t)
    }

    #[cfg(feature = "serialize")]
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
