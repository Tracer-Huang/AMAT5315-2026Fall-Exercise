use crate::{Args, Mode};
use rand::{Rng, rngs::StdRng};
use serde_json::{Value, json};
use std::collections::HashMap;

// A row cut retains one carry for each column and each intersecting diagonal.
// Selecting one square satisfies the row carry; all other x indices are zero.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
struct Boundary {
    columns: u32,
    left: u32,
    right: u32,
}

impl Boundary {
    fn choices(self, board: u32) -> impl Iterator<Item = (u32, Self)> {
        let mut free = board & !(self.columns | self.left | self.right);
        std::iter::from_fn(move || {
            if free == 0 {
                return None;
            }
            let column = 1 << free.trailing_zeros();
            free ^= column;
            Some((
                column.trailing_zeros(),
                Self {
                    columns: self.columns | column,
                    left: ((self.left | column) << 1) & board,
                    right: (self.right | column) >> 1,
                },
            ))
        })
    }
}

fn contract(state: Boundary, board: u32, counts: &mut HashMap<Boundary, u64>) -> u64 {
    if state.columns == board {
        return 1;
    }
    if let Some(&count) = counts.get(&state) {
        return count;
    }
    let count = state
        .choices(board)
        .map(|(_, next)| contract(next, board, counts))
        .sum();
    counts.insert(state, count);
    count
}

pub fn run(n: usize, args: &Args, rng: &mut StdRng) -> Result<(Value, Value), String> {
    if !(1..=12).contains(&n) {
        return Err("--n must be in 1..=12".into());
    }
    if matches!(args.mode, Mode::Plan) {
        return Ok((
            json!({"board_size":n,"max_boundary_bits":3*n}),
            json!({"plan":"sparse row cuts"}),
        ));
    }
    let board = (1 << n) - 1;
    let start = Boundary::default();
    let mut counts = HashMap::new();
    let total = contract(start, board, &mut counts);
    let mut samples = Vec::new();
    if matches!(args.mode, Mode::Sample) {
        if total == 0 {
            return Err("no valid boards to sample".into());
        }
        for _ in 0..args.samples.unwrap() {
            let mut state = start;
            let mut sample = Vec::new();
            while state.columns != board {
                let mut draw = rng.random_range(0..counts[&state]);
                for (column, next) in state.choices(board) {
                    let weight = if next.columns == board {
                        1
                    } else {
                        counts[&next]
                    };
                    if draw < weight {
                        sample.push(column);
                        state = next;
                        break;
                    }
                    draw -= weight;
                }
            }
            samples.push(sample);
        }
    }
    Ok((
        json!({"count":total,"stored_boundaries":counts.len()}),
        json!({"boards":samples}),
    ))
}
