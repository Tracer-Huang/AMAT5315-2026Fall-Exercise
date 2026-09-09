use md::periodic::{self, CellList};

fn assert_equivalent(positions: &[[f64; 2]], box_size: [f64; 2], cells: &mut CellList) {
    let mut naive = vec![[0.0; 2]; positions.len()];
    let mut optimized = naive.clone();
    let u_naive = periodic::forces(positions, box_size, &mut naive).unwrap();
    let u_cells = cells.forces(positions, &mut optimized).unwrap();
    assert!(
        (u_naive - u_cells).abs() < 1e-10 * u_naive.abs().max(1.0),
        "energy: {u_naive} vs {u_cells}"
    );
    for (a, b) in naive.iter().zip(&optimized) {
        for axis in 0..2 {
            assert!(
                (a[axis] - b[axis]).abs() < 1e-10 * a[axis].abs().max(1.0),
                "force mismatch {a:?} vs {b:?}"
            );
        }
    }
}

#[test]
fn cells_and_naive_agree_on_perturbed_lattices_across_boundaries() {
    for n in [100, 400, 1600] {
        let (mut positions, box_size) = periodic::lattice(n, 0.8).unwrap();
        let mut cells = CellList::new(box_size).unwrap();
        for iteration in 0..3 {
            for (i, x) in positions.iter_mut().enumerate() {
                x[0] = (x[0] + 0.025 * ((i + iteration) as f64).sin()).rem_euclid(box_size[0]);
                x[1] = (x[1] + 0.025 * ((2 * i + iteration) as f64).cos()).rem_euclid(box_size[1]);
            }
            assert_equivalent(&positions, box_size, &mut cells);
        }
    }
}

#[test]
fn two_cell_wide_box_deduplicates_wrapped_neighbour_cells() {
    let (mut positions, box_size) = periodic::lattice(36, 0.8).unwrap();
    assert_eq!((box_size[0] / 2.5).floor(), 2.0);
    assert_eq!((box_size[1] / 2.5).floor(), 2.0);
    positions[0] = [0.03, 0.07];
    let mut cells = CellList::new(box_size).unwrap();
    assert_equivalent(&positions, box_size, &mut cells);
    // Repeat with moved atoms so stale membership cannot pass.
    for p in &mut positions {
        p[0] = (p[0] + 1.8).rem_euclid(box_size[0]);
    }
    assert_equivalent(&positions, box_size, &mut cells);
}

#[test]
fn cells_preserve_cutoff_and_periodic_pair_cases() {
    let box_size = [6.0, 6.0];
    let mut cells = CellList::new(box_size).unwrap();
    for r in [2.5 - 1e-8, 2.5, 2.5 + 1e-8] {
        assert_equivalent(&[[0.1, 0.1], [0.1 + r, 0.1]], box_size, &mut cells);
    }
    assert_equivalent(&[[0.1, 0.1], [5.0, 0.1], [3.0, 3.0]], box_size, &mut cells);
    assert!(
        CellList::new([1e100, 1e100]).is_err(),
        "oversized grids must fail before allocation"
    );
}
