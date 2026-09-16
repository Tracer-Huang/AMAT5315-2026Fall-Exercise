use ising::Lattice;
fn oracle(l: usize, s: &[i8]) -> i64 {
    (0..l * l)
        .map(|i| {
            -(s[i] as i64) * (s[(i / l) * l + (i % l + 1) % l] as i64 + s[(i + l) % (l * l)] as i64)
        })
        .sum()
}
#[test]
fn ordered_energy() {
    let a = Lattice::new(4);
    assert_eq!(a.energy, -32);
    assert_eq!(a.magnet, 16);
}
#[test]
fn exhaustive_local_changes() {
    for l in [2, 3] {
        for bits in 0..(1usize << (l * l)) {
            let s: Vec<i8> = (0..l * l)
                .map(|i| if bits & (1 << i) == 0 { -1 } else { 1 })
                .collect();
            for i in 0..l * l {
                let mut a = Lattice::from_spins(l, s.clone());
                let before = oracle(l, &s);
                let mut t = s.clone();
                t[i] *= -1;
                assert_eq!(a.energy, before);
                assert_eq!(a.delta(i), oracle(l, &t) - before);
                a.flip(i);
                assert_eq!(a.energy, oracle(l, &t));
                assert_eq!(a.magnet, t.iter().map(|&x| x as i64).sum());
            }
        }
    }
}
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[test]
fn metropolis_infinite_temperature_accepts_every_proposal() {
    let mut a = Lattice::new(4);
    let mut r = ChaCha8Rng::seed_from_u64(3);
    assert_eq!(a.metropolis(&mut r, &[1.; 5]), 16);
    assert_eq!(a.energy, oracle(4, &a.spins));
}
#[test]
fn wolff_unit_bond_flips_whole_ordered_lattice() {
    let mut a = Lattice::new(4);
    let mut r = ChaCha8Rng::seed_from_u64(3);
    assert_eq!(a.wolff(&mut r, 1.), 16);
    assert_eq!(a.magnet, -16);
    assert_eq!(a.energy, -32);
}
#[test]
fn metropolis_detailed_balance_exact_two_by_two() {
    let t = 2.3;
    let table = ising::acceptance(t);
    for bits in 0..16 {
        let a = Lattice::from_spins(
            2,
            (0..4)
                .map(|i| if bits & (1 << i) == 0 { -1 } else { 1 })
                .collect(),
        );
        for i in 0..4 {
            let d = a.delta(i);
            let forward = (-((a.energy) as f64) / t).exp() * table[((d + 8) / 4) as usize];
            let reverse = (-((a.energy + d) as f64) / t).exp() * table[((-d + 8) / 4) as usize];
            assert!((forward - reverse).abs() < 1e-10);
        }
    }
}

#[test]
fn wolff_zero_bond_is_one_flip_and_energy_stays_exact() {
    let mut a = Lattice::new(3);
    let mut r = ChaCha8Rng::seed_from_u64(2026);
    for _ in 0..100 {
        assert_eq!(a.wolff(&mut r, 0.), 1);
        assert_eq!(a.energy, oracle(3, &a.spins));
    }
    for _ in 0..100 {
        a.wolff(&mut r, 0.63);
        assert_eq!(a.energy, oracle(3, &a.spins));
    }
}
