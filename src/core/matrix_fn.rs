use crate::core::Matrix;

impl Matrix {
    pub fn add(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape != rhs.shape {
            Err(format!(
                "size {} is not compatible with size {}",
                rhs.shape, self.shape
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 0..self.shape.row * rhs.shape.col {
                m.data[i] = self.data[i] + rhs.data[i];
            }
            Ok(m)
        }
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape.col != rhs.shape.row {
            Err(format!(
                "size {} is not compatible with size {}",
                rhs.shape, self.shape
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 1..=self.shape.col {
                m = m.add(&Self::outter(self.get_col(i)?, rhs.get_row(i)?)?)?;
            }
            Ok(m)
        }
    }

    pub fn smul(&self, k: f64) -> Result<Self, String> {
        let mut m = Self::zeros(self.shape.row, self.shape.col)?;
        for i in 0..self.data.len() {
            m.data[i] = self.data[i] * k;
        }
        Ok(m)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape != rhs.shape {
            Err(format!(
                "size {} is not compatible with size {}",
                rhs.shape, self.shape
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 0..self.shape.row * rhs.shape.col {
                m.data[i] = self.data[i] - rhs.data[i];
            }
            Ok(m)
        }
    }
}
