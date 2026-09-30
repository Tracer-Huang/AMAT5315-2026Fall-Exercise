#[derive(Clone, Debug)]
pub enum Action {
    Restore(usize),
    Call(usize),
    Store(usize),
    Grad(usize),
    Fetch(usize),
}
pub fn schedule(n: usize, budget: usize) -> Vec<Action> {
    assert!(n > 0 && budget > 0);
    fn capacity(d: usize, t: usize) -> u128 {
        (1..=d).fold(1u128, |v, k| {
            v.saturating_mul((t + k) as u128) / (k as u128)
        })
    }
    fn split(s: usize, e: usize, d: usize, t: usize) -> usize {
        let mut k = (d * s + t * e).div_ceil(d + t);
        if k >= e && d > 0 {
            k = (s + 1).max(e - 1)
        }
        k
    }
    fn visit(b: usize, s: usize, mut e: usize, mut d: usize, mut t: usize, a: &mut Vec<Action>) {
        if s > b {
            d -= 1;
            a.push(Action::Restore(b));
            for i in b..s {
                a.push(Action::Call(i));
            }
            a.push(Action::Store(s));
        }
        let mut k = split(s, e, d, t);
        while t > 0 && k < e {
            visit(s, k, e, d, t, a);
            t -= 1;
            e = k;
            k = split(s, e, d, t);
        }
        a.push(Action::Grad(s));
        if s > b {
            a.push(Action::Fetch(s));
        }
    }
    let mut t = 1;
    while capacity(budget, t) < n as u128 {
        t += 1;
    }
    let mut a = vec![];
    visit(0, 0, n, budget, t, &mut a);
    a
}
unsafe extern "C" {
    fn enzyme_cube(x: f64, out: *mut f64);
    fn wave_primal(
        s: *const f64,
        c: *const f64,
        q: *const f64,
        sigma: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        out: *mut f64,
    );
    fn wave_jvp(
        s: *const f64,
        ds: *const f64,
        c: *const f64,
        dc: *const f64,
        q: *const f64,
        sigma: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        out: *mut f64,
        dout: *mut f64,
    );
    fn wave_vjp(
        s: *const f64,
        ds: *mut f64,
        c: *const f64,
        dc: *mut f64,
        q: *const f64,
        sigma: *const f64,
        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,
        out: *mut f64,
        dout: *mut f64,
    );
}
pub fn cube_probe(x: f64) -> [f64; 3] {
    let mut a = [0.; 3];
    unsafe {
        enzyme_cube(x, a.as_mut_ptr());
    }
    a
}
#[derive(Clone)]
pub struct Model {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub c: Vec<f64>,
    pub sigma: Vec<f64>,
}
impl Model {
    pub fn size(&self) -> usize {
        self.nx * self.nz
    }
    fn check(&self, s: &[f64], q: &[f64], out: &[f64]) {
        assert!(self.nx >= 3 && self.nz >= 3);
        let m = self.size();
        assert_eq!(s.len(), 2 * m);
        assert_eq!(out.len(), 2 * m);
        assert_eq!(q.len(), m);
        assert_eq!(self.c.len(), m);
        assert_eq!(self.sigma.len(), m);
    }
    pub fn forward(&self, s: &[f64], q: &[f64], out: &mut [f64]) {
        self.check(s, q, out);
        unsafe {
            wave_primal(
                s.as_ptr(),
                self.c.as_ptr(),
                q.as_ptr(),
                self.sigma.as_ptr(),
                self.nx,
                self.nz,
                self.dx,
                self.dt,
                out.as_mut_ptr(),
            );
        }
    }
    pub fn jvp(
        &self,
        s: &[f64],
        ds: &[f64],
        dc: &[f64],
        q: &[f64],
        out: &mut [f64],
        dout: &mut [f64],
    ) {
        self.check(s, q, out);
        assert_eq!(ds.len(), s.len());
        assert_eq!(dc.len(), self.size());
        assert_eq!(dout.len(), s.len());
        unsafe {
            wave_jvp(
                s.as_ptr(),
                ds.as_ptr(),
                self.c.as_ptr(),
                dc.as_ptr(),
                q.as_ptr(),
                self.sigma.as_ptr(),
                self.nx,
                self.nz,
                self.dx,
                self.dt,
                out.as_mut_ptr(),
                dout.as_mut_ptr(),
            );
        }
    }
    pub fn vjp(&self, s: &[f64], q: &[f64], weights: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut out = vec![0.; s.len()];
        self.check(s, q, &out);
        assert_eq!(weights.len(), s.len());
        let mut ds = vec![0.; s.len()];
        let mut dc = vec![0.; self.size()];
        let mut w = weights.to_vec();
        unsafe {
            wave_vjp(
                s.as_ptr(),
                ds.as_mut_ptr(),
                self.c.as_ptr(),
                dc.as_mut_ptr(),
                q.as_ptr(),
                self.sigma.as_ptr(),
                self.nx,
                self.nz,
                self.dx,
                self.dt,
                out.as_mut_ptr(),
                w.as_mut_ptr(),
            );
        }
        (ds, dc)
    }
}
#[allow(clippy::too_many_arguments)]
pub fn step(
    prev: &[f64],
    cur: &[f64],
    c: &[f64],
    sigma: &[f64],
    q: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
) -> Vec<f64> {
    let model = Model {
        nx,
        nz,
        dx,
        dt,
        c: c.to_vec(),
        sigma: sigma.to_vec(),
    };
    let s = [prev, cur].concat();
    let mut out = vec![0.; s.len()];
    model.forward(&s, q, &mut out);
    out[nx * nz..].to_vec()
}
