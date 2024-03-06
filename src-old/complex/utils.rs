use crate::{complex::Complex, error::RMatrixError};

impl Default for Complex {
    fn default() -> Self {
        Complex::new(f64::default(), f64::default())
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}{:+.3}I", self.re(), self.im())
    }
}

impl std::str::FromStr for Complex {
    type Err = Box<dyn std::error::Error>;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let v = value.trim();
        if v.is_empty() && !v.ends_with('I') {
            Err(Box::new(RMatrixError::ParseFailed(value.to_owned())))
        } else {
            let mut v = v.chars();
            let mut re: Vec<char> = Vec::new();
            re.push(v.next().unwrap_or('0'));
            let mut im: Vec<char> = Vec::new();
            let mut is_re: bool = true;
            v.for_each(|c| {
                if is_re && !(c == '+' || c == '-') {
                    re.push(c);
                } else {
                    if is_re {
                        is_re = false;
                    }
                    im.push(c)
                }
            });
            Ok(Complex(
                re.iter()
                    .filter(|c| !c.is_whitespace())
                    .collect::<String>()
                    .parse()?,
                if im.is_empty() {
                    f64::default()
                } else {
                    im[..im.len() - 1]
                        .iter()
                        .filter(|c| !c.is_whitespace())
                        .collect::<String>()
                        .parse()
                        .unwrap_or_default()
                },
            ))
        }
    }
}

impl std::iter::Sum for Complex {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut n = Complex::default();
        for i in iter {
            n = n + i;
        }
        n
    }
}
