#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

#[autodiff_forward(cube_forward, Dual, Dual)]
#[autodiff_reverse(cube_reverse, Active, Active)]
fn cube(x: f64) -> f64 {
    x * x * x
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
    let (y, dy) = cube_forward(x, 1.);
    let (_, adj) = cube_reverse(x, 1.);
    unsafe {
        *out = y;
        *out.add(1) = dy;
        *out.add(2) = adj;
    }
}
unsafe extern "C" {
    fn abort() -> !;
}
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}

#[autodiff_forward(
    advance_fwd,
    Dual,
    Dual,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Dual
)]
#[autodiff_reverse(
    advance_rev,
    Duplicated,
    Duplicated,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Duplicated
)]
fn advance(
    s: &[f64],
    c: &[f64],
    q: &[f64],
    sigma: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    out: &mut [f64],
) {
    let m = nx * nz;
    for z in 0..nz {
        for x in 0..nx {
            let i = z * nx + x;
            if x == 0 || z == 0 || x + 1 == nx || z + 1 == nz {
                out[i] = 0.;
                out[m + i] = 0.;
            } else {
                let u = s[m + i];
                let lap = (s[m + i - 1] + s[m + i + 1] + s[m + i - nx] + s[m + i + nx] - 4. * u)
                    / (dx * dx);
                out[i] = u;
                out[m + i] = (2. * u - (1. - sigma[i] * dt) * s[i]
                    + dt * dt * (c[i] * c[i] * lap + q[i]))
                    / (1. + sigma[i] * dt);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_primal(
    s: *const f64,
    c: *const f64,
    q: *const f64,
    sigma: *const f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    out: *mut f64,
) {
    unsafe {
        let m = nx * nz;
        advance(
            core::slice::from_raw_parts(s, 2 * m),
            core::slice::from_raw_parts(c, m),
            core::slice::from_raw_parts(q, m),
            core::slice::from_raw_parts(sigma, m),
            nx,
            nz,
            dx,
            dt,
            core::slice::from_raw_parts_mut(out, 2 * m),
        );
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_jvp(
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
) {
    unsafe {
        let m = nx * nz;
        advance_fwd(
            core::slice::from_raw_parts(s, 2 * m),
            core::slice::from_raw_parts(ds, 2 * m),
            core::slice::from_raw_parts(c, m),
            core::slice::from_raw_parts(dc, m),
            core::slice::from_raw_parts(q, m),
            core::slice::from_raw_parts(sigma, m),
            nx,
            nz,
            dx,
            dt,
            core::slice::from_raw_parts_mut(out, 2 * m),
            core::slice::from_raw_parts_mut(dout, 2 * m),
        );
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wave_vjp(
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
) {
    unsafe {
        let m = nx * nz;
        advance_rev(
            core::slice::from_raw_parts(s, 2 * m),
            core::slice::from_raw_parts_mut(ds, 2 * m),
            core::slice::from_raw_parts(c, m),
            core::slice::from_raw_parts_mut(dc, m),
            core::slice::from_raw_parts(q, m),
            core::slice::from_raw_parts(sigma, m),
            nx,
            nz,
            dx,
            dt,
            core::slice::from_raw_parts_mut(out, 2 * m),
            core::slice::from_raw_parts_mut(dout, 2 * m),
        );
    }
}
