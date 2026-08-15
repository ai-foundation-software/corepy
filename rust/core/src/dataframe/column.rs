#[derive(Clone, Debug, PartialEq)]
pub enum Column {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<String>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::String(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
