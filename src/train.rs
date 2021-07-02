use rand::Rng;

use crate::dna::{Dimension, CHRO_LEN, DNA};

pub struct Trainer<T>
where
    T: Fn(f64, f64) -> f64,
{
    target_func: T,
    _initialized: bool,
    domain: ((f64, f64), (f64, f64)),
    population: usize,
    mutation_rate: f64,
    obsolete_rate: f64,
}

pub struct TrainerConfig {
    pub domain: ((f64, f64), (f64, f64)),
    pub population: usize,
    pub mutation_rate: f64,
    pub obsolete_rate: f64,
}

impl<T> Trainer<T>
where
    T: Fn(f64, f64) -> f64,
{
    pub fn new(target_func: T) -> Self {
        Trainer {
            target_func,
            _initialized: false,
            domain: ((0., 0.), (0., 0.)),
            population: 0,
            mutation_rate: 0.,
            obsolete_rate: 0.,
        }
    }

    pub fn init(&mut self, config: TrainerConfig) {
        self.domain = config.domain;
        self.population = config.population;
        self.mutation_rate = config.mutation_rate;
        self.obsolete_rate = config.obsolete_rate;
        self._initialized = true;
    }

    fn train(&mut self, epoches: usize) {
        let mut fitness_list = vec![];
        let mut rng = rand::thread_rng();
        let population = self.population;
        // Vec<(DNA, ID标识)>
        let mut individuals = (0..population)
            .map(|_| (DNA::random(), rng.gen::<f64>()))
            .collect::<Vec<_>>();

        for epoch in 0..epoches {
            // 初始化fitness_list，保存所有个体的适应度信息
            // 按照fitness从高到低进行排序，然后淘汰掉部分个体
            fitness_list.clear();
            for (i, (individual, _)) in individuals.iter().enumerate() {
                let (x, y) = individual.get_value(self.domain);
                let fitness = (self.target_func)(x, y);
                if fitness > 0. {
                    // fitness_list排序过后顺序与individuals不一致
                    // 保存i索引，方便找到对应的individual
                    fitness_list.push((i, fitness));
                }
            }
            fitness_list.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("NaN"));
            for _ in 0..(fitness_list.len() as f64 * self.obsolete_rate) as usize {
                fitness_list.pop();
            }
            println!(
                "{}th eopch({}%): {}",
                epoch + 1,
                (fitness_list.len() * 100) as f64 / population as f64,
                fitness_list[0].1
            );

            // 按照fitness累加并除以总和生成概率表，然后依照概率挑选下一轮个体
            let total_socre = fitness_list.iter().fold(0., |acc, (_, x)| acc + x);
            let mut picked_rates = vec![];
            fitness_list.iter().fold(0., |acc, (_, x)| {
                picked_rates.push((acc + x) / total_socre);
                acc + x
            });
            let mut new_individuals = Vec::with_capacity(self.population);
            for _ in 0..population {
                let tmp = rng.gen::<f64>();
                for i in 0..picked_rates.len() {
                    if tmp <= picked_rates[i] {
                        new_individuals.push(individuals[fitness_list[i].0].clone());
                        break;
                    }
                }
            }

            // 交换DNA，执行变异操作
            for i in 0..(population / 2) {
                // 如果执行交叉操作的是两个相同的DNA，则以变异操作取代
                if new_individuals[i * 2].1 == new_individuals[i * 2 + 1].1 {
                    new_individuals[i * 2]
                        .0
                        .mutate((rng.gen_range(0..CHRO_LEN), rng.gen_range(0..CHRO_LEN)));
                    new_individuals[i * 2].1 = rng.gen();
                    new_individuals[i * 2 + 1]
                        .0
                        .mutate((rng.gen_range(0..CHRO_LEN), rng.gen_range(0..CHRO_LEN)));
                    new_individuals[i * 2 + 1].1 = rng.gen();
                } else {
                    let mut fragment = new_individuals[i * 2 + 1]
                        .0
                        .get_fragment(rng.gen_range(1..CHRO_LEN), Dimension::X);
                    fragment = new_individuals[i * 2].0.exchange(fragment, Dimension::X);
                    new_individuals[i * 2 + 1]
                        .0
                        .exchange(fragment, Dimension::X);
                    fragment = new_individuals[i * 2 + 1]
                        .0
                        .get_fragment(rng.gen_range(1..CHRO_LEN), Dimension::Y);
                    fragment = new_individuals[i * 2].0.exchange(fragment, Dimension::Y);
                    new_individuals[i * 2 + 1]
                        .0
                        .exchange(fragment, Dimension::Y);
                }
            }
            for i in 0..population {
                if rng.gen::<f64>() <= self.mutation_rate {
                    new_individuals[i]
                        .0
                        .mutate((rng.gen_range(0..CHRO_LEN), rng.gen_range(0..CHRO_LEN)))
                }
            }
            // 本轮最优个体保持不变
            new_individuals.pop();
            new_individuals.insert(0, individuals[fitness_list[0].0].clone());
            individuals = new_individuals;
        }

        println!(
            "best individual: {:?} => {}",
            individuals[fitness_list[0].0].0.get_value(self.domain),
            fitness_list[0].1
        );
    }

    pub fn start(&mut self, epoches: usize) {
        println!("start training {} epoches", epoches);
        self.train(epoches);
    }
}
