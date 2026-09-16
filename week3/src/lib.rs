use rand::Rng;
pub struct Lattice {
    pub l: usize,
    pub spins: Vec<i8>,
    pub energy: i64,
    pub magnet: i64,
    neighbors: Vec<[usize; 4]>,
    marks: Vec<u64>,
    generation: u64,
    cluster: Vec<usize>,
}
impl Lattice {
    pub fn new(l: usize) -> Self {
        Self::from_spins(l, vec![1; l * l])
    }
    pub fn from_spins(l: usize, spins: Vec<i8>) -> Self {
        assert!(l >= 2 && spins.len() == l * l && spins.iter().all(|s| s.abs() == 1));
        let neighbors = (0..l * l)
            .map(|i| {
                let (r, c) = (i / l, i % l);
                [
                    r * l + (c + 1) % l,
                    r * l + (c + l - 1) % l,
                    ((r + 1) % l) * l + c,
                    ((r + l - 1) % l) * l + c,
                ]
            })
            .collect::<Vec<_>>();
        let energy = (0..l * l)
            .map(|i| {
                -(spins[i] as i64) * (spins[neighbors[i][0]] as i64 + spins[neighbors[i][2]] as i64)
            })
            .sum();
        let magnet = spins.iter().map(|&s| s as i64).sum();
        Self {
            l,
            spins,
            energy,
            magnet,
            neighbors,
            marks: vec![0; l * l],
            generation: 0,
            cluster: Vec::with_capacity(l * l),
        }
    }
    pub fn delta(&self, i: usize) -> i64 {
        2 * self.spins[i] as i64
            * self.neighbors[i]
                .iter()
                .map(|&j| self.spins[j] as i64)
                .sum::<i64>()
    }
    pub fn flip(&mut self, i: usize) {
        self.energy += self.delta(i);
        self.magnet -= 2 * self.spins[i] as i64;
        self.spins[i] *= -1;
    }
    pub fn metropolis<R: Rng>(&mut self, rng: &mut R, table: &[f64; 5]) -> usize {
        let mut accepted = 0;
        for _ in 0..self.spins.len() {
            let i = rng.gen_range(0..self.spins.len());
            let d = self.delta(i);
            if d <= 0 || rng.gen::<f64>() < table[((d + 8) / 4) as usize] {
                self.flip(i);
                accepted += 1;
            }
        }
        accepted
    }
    pub fn wolff<R: Rng>(&mut self, rng: &mut R, p: f64) -> usize {
        self.generation += 1;
        self.cluster.clear();
        let seed = rng.gen_range(0..self.spins.len());
        let sign = self.spins[seed];
        self.cluster.push(seed);
        self.marks[seed] = self.generation;
        let mut head = 0;
        while head < self.cluster.len() {
            let i = self.cluster[head];
            for j in self.neighbors[i] {
                if self.marks[j] != self.generation && self.spins[j] == sign && rng.gen::<f64>() < p
                {
                    self.marks[j] = self.generation;
                    self.cluster.push(j);
                }
            }
            head += 1;
        }
        // Sequential local flips telescope to the exact energy change, including L=2 parallel bonds.
        for k in 0..self.cluster.len() {
            self.flip(self.cluster[k]);
        }
        self.cluster.len()
    }
}
pub fn acceptance(t: f64) -> [f64; 5] {
    [-8., -4., 0., 4., 8.].map(|d: f64| (-d / t).exp().min(1.))
}
