use week10::boundary;
fn exact(l:usize,k:f64)->f64 { let mut z=0.;for state in 0..(1usize<<(l*l)){let mut sum=0.;for y in 0..l {for x in 0..l {let s=if state>>(y*l+x)&1==1 {1.}else{-1.};if x+1<l {sum+=s*if state>>(y*l+x+1)&1==1 {1.}else{-1.};} if y+1<l {sum+=s*if state>>((y+1)*l+x)&1==1 {1.}else{-1.};}}}z+=(k*sum).exp();}z.ln()/(l*l) as f64}
#[test] fn open_square_matches_enumeration(){for k in [0.,0.44,0.7] {let got=boundary(4,k,8).0;assert!((got-exact(4,k)).abs()<1e-8,"k={k} got={got} exact={}",exact(4,k));}}
fn exact_periodic(l:usize,k:f64)->f64 {let mut z=0.;for state in 0..(1usize<<(l*l)){let mut e=0.;for y in 0..l {for x in 0..l {let spin=|i:usize|if state>>i&1==1 {1.} else {-1.};let s=spin(y*l+x);e+=s*spin(y*l+(x+1)%l)+s*spin(((y+1)%l)*l+x);}} z+=(k*e).exp();}z.ln()/(l*l)as f64}
#[test] fn periodic_matches_enumeration(){for (l,chi,tol) in [(2,16,1e-8),(4,8,2e-3)] {let got=week10::trg(l,0.44,chi).0;let ex=exact_periodic(l,0.44);assert!((got-ex).abs()<tol,"l={l} got={got} exact={ex}");}}
#[test] fn open_16_against_independent_transfer(){let got=week10::boundary(16,0.485,16).0;assert!((got-0.9654522176646275).abs()<1e-9,"got={got}");}
