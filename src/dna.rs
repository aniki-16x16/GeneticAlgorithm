use rand::{self, Rng};

pub const CHRO_LEN: usize = 32;
const CHRO_MAX: f64 = 4294967295.;

#[derive(Debug, Clone)]
pub struct Chromosome([u8; CHRO_LEN]);

impl Chromosome {
    pub fn random() -> Self {
        let mut list = [0u8; CHRO_LEN];
        let mut rng = rand::thread_rng();
        for i in 0..CHRO_LEN {
            if rng.gen::<f64>() >= 0.5 {
                list[i] = 1u8;
            }
        }
        Chromosome(list)
    }

    pub fn get_value(&self) -> f64 {
        self.0
            .iter()
            .rev()
            .enumerate()
            .fold(0., |acc, (i, v)| match v {
                0 => acc,
                1 => acc + 2f64.powi(i as i32),
                _ => unreachable!(),
            })
            / CHRO_MAX
    }

    pub fn replace_by_slice(&mut self, anchor: usize, slice: &[u8]) -> Vec<u8> {
        let mut result = vec![];
        for i in 0..slice.len() {
            result.push(self.0[anchor + i]);
            self.0[anchor + i] = slice[i];
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct DNA(Chromosome, Chromosome);

pub enum Dimension {
    X,
    Y,
}

impl DNA {
    pub fn random() -> Self {
        DNA(Chromosome::random(), Chromosome::random())
    }

    pub fn get_value(&self, range: ((f64, f64), (f64, f64))) -> (f64, f64) {
        (
            self.0.get_value() * (range.0 .1 - range.0 .0) + range.0 .0,
            self.1.get_value() * (range.1 .1 - range.1 .0) + range.1 .0,
        )
    }

    pub fn get_fragment(&self, anchor: usize, dimension: Dimension) -> Vec<u8> {
        match dimension {
            Dimension::X => (&self.0 .0[..anchor]).to_vec(),
            Dimension::Y => (&self.1 .0[..anchor]).to_vec(),
        }
    }

    pub fn exchange(&mut self, fragment: Vec<u8>, dimension: Dimension) -> Vec<u8> {
        match dimension {
            Dimension::X => self.0.replace_by_slice(0, &fragment),
            Dimension::Y => self.1.replace_by_slice(0, &fragment),
        }
    }

    pub fn mutate(&mut self, anchors: (usize, usize)) {
        match self.0 .0[anchors.0] {
            0 => self.0 .0[anchors.0] = 1,
            1 => self.0 .0[anchors.0] = 0,
            _ => unreachable!(),
        };
        match self.1 .0[anchors.1] {
            0 => self.1 .0[anchors.1] = 1,
            1 => self.1 .0[anchors.1] = 0,
            _ => unreachable!(),
        };
    }
}
