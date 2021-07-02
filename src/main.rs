use std::f64::consts;
use train::{Trainer, TrainerConfig};

mod dna;
mod train;

const DOMAIN: ((f64, f64), (f64, f64)) = ((-consts::PI, consts::PI), (-consts::PI, consts::PI));

fn main() {
    println!("find the maximum of");
    println!("z(x, y) = sin(π * x^2) - cos(π * y^2 - x) + x^2 * y - y^2 * x");
    println!("x ∈ [-π, π]; y ∈ [-π, π]\n");
    let target_func = |x: f64, y: f64| -> f64 {
        f64::sin(x * x * consts::PI) - f64::cos(y * y * consts::PI - x) + x * x * y - y * y * x
    };

    let mut trainer = Trainer::new(target_func);
    trainer.init(TrainerConfig {
        domain: DOMAIN,
        population: 100,
        mutation_rate: 0.01,
        obsolete_rate: 0.05,
    });
    trainer.start(100);
}
