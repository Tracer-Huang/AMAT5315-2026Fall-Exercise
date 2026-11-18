use serde_json::{Value, json};

fn energy(gate: &str, a: i32, b: i32, z: i32, w: i32) -> i32 {
    match gate {
        "AND" => a * b - 2 * a * z - 2 * b * z - a - b + 2 * z,
        "OR" => a * b - 2 * a * z - 2 * b * z + a + b - 2 * z,
        "NOT" => a * z,
        "COPY" => -a * z,
        "SET0" => z,
        "SET1" => -z,
        "XOR" => a * b - a * z - 2 * a * w - b * z - 2 * b * w + 2 * z * w - a - b + z + 2 * w,
        _ => unreachable!(),
    }
}
fn spin(bit: i32) -> i32 {
    2 * bit - 1
}

pub fn run() -> (Value, Value) {
    let mut gates = serde_json::Map::new();
    for (gate, inputs) in [
        ("AND", 2),
        ("OR", 2),
        ("NOT", 1),
        ("COPY", 1),
        ("SET0", 0),
        ("SET1", 0),
        ("XOR", 2),
    ] {
        let mut rows = Vec::new();
        for mask in 0..1 << inputs {
            let bits: Vec<_> = (0..inputs)
                .map(|i| (mask >> (inputs - 1 - i)) & 1)
                .collect();
            let a = *bits.first().unwrap_or(&0);
            let b = *bits.get(1).unwrap_or(&0);
            for z in 0..2 {
                let expected = match gate {
                    "AND" => a & b,
                    "OR" => a | b,
                    "NOT" => 1 - a,
                    "COPY" => a,
                    "SET0" => 0,
                    "SET1" => 1,
                    "XOR" => a ^ b,
                    _ => unreachable!(),
                };
                let e = [-1, 1].map(|w| energy(gate, spin(a), spin(b), spin(z), w));
                rows.push(
                    json!({"inputs":bits,"output":z,"allowed":z==expected,"energy":e[0].min(e[1])}),
                );
            }
        }
        gates.insert(gate.into(), json!(rows));
    }
    let mut circuit = Vec::new();
    for mask in 0..16 {
        let [a, b, t, out] = [3, 2, 1, 0].map(|i| (mask >> i) & 1);
        let e = energy("AND", spin(a), spin(b), spin(t), 0)
            + energy("NOT", spin(t), 0, spin(out), 0)
            + energy("SET1", 0, 0, spin(out), 0);
        circuit.push(json!({"bits":[a,b,t,out],"energy":e,"allowed":t==a&b && out==1-t && out==1}));
    }
    (
        json!({"gates":7,"circuit_states":16}),
        json!({"gates":gates,"circuit":circuit}),
    )
}
