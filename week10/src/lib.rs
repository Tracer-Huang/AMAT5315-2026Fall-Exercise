use std::os::raw::c_char;
#[link(name="Accelerate", kind="framework")]
unsafe extern "C" { fn dgesvd_(jobu:*const c_char,jobvt:*const c_char,m:*const i32,n:*const i32,a:*mut f64,lda:*const i32,s:*mut f64,u:*mut f64,ldu:*const i32,vt:*mut f64,ldvt:*const i32,work:*mut f64,lwork:*const i32,info:*mut i32); }
pub fn svd(a:&[f64],m:usize,n:usize)->(Vec<f64>,Vec<f64>,Vec<f64>){
 assert_eq!(a.len(),m*n);let q=m.min(n);let (mi,ni,qi)=(m as i32,n as i32,q as i32);let job=b'S' as c_char;let mut a=a.to_vec();let mut s=vec![0.;q];let mut u=vec![0.;m*q];let mut vt=vec![0.;q*n];let mut info=0;let mut query=0.;let lw=-1;
 unsafe {dgesvd_(&job,&job,&mi,&ni,a.as_mut_ptr(),&mi,s.as_mut_ptr(),u.as_mut_ptr(),&mi,vt.as_mut_ptr(),&qi,&mut query,&lw,&mut info);}assert_eq!(info,0);let lw=query as i32;let mut work=vec![0.;lw as usize];
 unsafe {dgesvd_(&job,&job,&mi,&ni,a.as_mut_ptr(),&mi,s.as_mut_ptr(),u.as_mut_ptr(),&mi,vt.as_mut_ptr(),&qi,work.as_mut_ptr(),&lw,&mut info);}assert_eq!(info,0,"SVD failed {m}x{n}");(u,s,vt)
}

#[derive(Clone,Debug)]
pub struct Site { pub left:usize, pub right:usize, pub data:Vec<f64> }
impl Site {fn new(l:usize,r:usize)->Self {Self{left:l,right:r,data:vec![0.;l*2*r]}} fn idx(&self,l:usize,p:usize,r:usize)->usize {(l*2+p)*self.right+r} fn get(&self,l:usize,p:usize,r:usize)->f64 {self.data[self.idx(l,p,r)]} fn set(&mut self,l:usize,p:usize,r:usize,v:f64){let i=self.idx(l,p,r);self.data[i]=v;}}
/// Returns g, final boundary MPS, normalization log-scales, discarded squared weights.
pub fn boundary(l:usize,k:f64,chi:usize)->(f64,Vec<Site>,Vec<f64>,Vec<f64>){
 assert!(l>=2 && k>=0. && chi>=1);let c=k.cosh().sqrt();let s=k.sinh().sqrt();let w=[[c,s],[c,-s]]; let b=1./c;
 let mut sites:Vec<Site>=(0..l).map(|_|{let mut a=Site::new(1,1);a.set(0,0,0,b);a}).collect();let mut scales=Vec::new();let mut discarded=Vec::new();
 for _row in 0..l {
  let mut next=Vec::new();
  for x in 0..l {let old=&sites[x];let dl=if x==0 {1}else{2};let dr=if x==l-1 {1}else{2};let mut t=Site::new(old.left*dl,old.right*dr);
   for a in 0..old.left {for z in 0..old.right {for hl in 0..dl {for hr in 0..dr {for d in 0..2 {let mut v=0.;for u in 0..2 {for spin in 0..2 {let left=if x==0 {1.}else{w[spin][hl]};let right=if x==l-1 {1.}else{w[spin][hr]};v+=old.get(a,u,z)*left*right*w[spin][u]*w[spin][d];}}t.set(a*dl+hl,d,z*dr+hr,v);}}}}}next.push(t);
  }
  sites=next;
  // Right-canonicalize without truncation.
  for i in (1..l).rev(){let t=&sites[i];let (a,r)=(t.left,t.right);let cols=2*r;let mut m=vec![0.;a*cols];for x in 0..a {for p in 0..2 {for y in 0..r {m[x+a*(p*r+y)]=t.get(x,p,y);}}}let (u,sv,vt)=svd(&m,a,cols);let q=sv.len();let mut right=Site::new(q,r);for z in 0..q{for p in 0..2{for y in 0..r{right.set(z,p,y,vt[z+q*(p*r+y)]);}}}let prev=&sites[i-1];let mut left=Site::new(prev.left,q);for x in 0..prev.left{for p in 0..2{for z in 0..q{let mut v=0.;for y in 0..a{v+=prev.get(x,p,y)*u[y+a*z]*sv[z];}left.set(x,p,z,v);}}}sites[i-1]=left;sites[i]=right;}
  // Left-canonicalize, truncating the Schmidt spectrum.
  for i in 0..l-1 {let t=&sites[i];let (a,r)=(t.left,t.right);let rows=a*2;let mut m=vec![0.;rows*r];for x in 0..a{for p in 0..2{for y in 0..r{m[x*2+p+rows*y]=t.get(x,p,y);}}}let (u,sv,vt)=svd(&m,rows,r);let full=sv.len();let q=chi.min(full);discarded.push(sv[q..].iter().map(|v|v*v).sum());let mut left=Site::new(a,q);for x in 0..a{for p in 0..2{for z in 0..q{left.set(x,p,z,u[x*2+p+rows*z]);}}}let next=&sites[i+1];let mut right=Site::new(q,next.right);for z in 0..q{for p in 0..2{for y in 0..next.right{let mut v=0.;for x in 0..r{v+=sv[z]*vt[z+full*x]*next.get(x,p,y);}right.set(z,p,y,v);}}}sites[i]=left;sites[i+1]=right;}
  let scale=sites[l-1].data.iter().fold(0f64,|a,v|a.max(v.abs()));assert!(scale>0.&&scale.is_finite());for v in &mut sites[l-1].data{*v/=scale;}scales.push(scale.ln());
 }
 let mut v=vec![1.];for t in &sites {let mut next=vec![0.;t.right];for a in 0..t.left{for r in 0..t.right{next[r]+=v[a]*t.get(a,0,r)*b;}}v=next;}assert!(v[0]>0.);let logz=scales.iter().sum::<f64>()+v[0].ln();(logz/(l*l) as f64,sites,scales,discarded)
}
#[link(name="Accelerate",kind="framework")]
unsafe extern "C" {fn dgemm_(ta:*const c_char,tb:*const c_char,m:*const i32,n:*const i32,k:*const i32,alpha:*const f64,a:*const f64,lda:*const i32,b:*const f64,ldb:*const i32,beta:*const f64,c:*mut f64,ldc:*const i32);}
#[derive(Debug)]
pub struct TrgStep {pub log_scale:f64,pub singular1:Vec<f64>,pub singular2:Vec<f64>,pub factors:Vec<Vec<f64>>,pub old_dim:usize,pub new_dim:usize}
/// Split matrix indices with axis order (l,u,r,d) and (d,l,u,r), regroup four halves.
pub fn trg(l:usize,k:f64,chi:usize)->(f64,Vec<f64>,Vec<TrgStep>){
 assert!(l>=2&&l.is_power_of_two()&&k>=0.&&chi>=1);let c=k.cosh().sqrt();let s=k.sinh().sqrt();let w=[[c,s],[c,-s]];let mut d=2;let mut a=vec![0.;16];for r in 0..2{for u in 0..2{for left in 0..2{for down in 0..2{a[((r*2+u)*2+left)*2+down]=(0..2).map(|sp|w[sp][r]*w[sp][u]*w[sp][left]*w[sp][down]).sum();}}}}
 let mut logscale=0.;let mut steps=Vec::new();let mut count=(l*l)as f64;
 for _ in 0..2*l.ilog2(){let scale=a.iter().fold(0f64,|x,v|x.max(v.abs()));assert!(scale>0.);for v in &mut a{*v/=scale;}logscale+=count*scale.ln();let m=d*d;let mut m1=vec![0.;m*m];let mut m2=vec![0.;m*m];for r in 0..d{for u in 0..d{for left in 0..d{for down in 0..d{let v=a[((r*d+u)*d+left)*d+down];m1[left*d+u+m*(r*d+down)]=v;m2[down*d+left+m*(u*d+r)]=v;}}}}
 let (u1,s1,v1)=svd(&m1,m,m);let (u2,s2,v2)=svd(&m2,m,m);let q=chi.min(m);let mut f1=vec![0.;d*d*q];let mut f2=f1.clone();let mut f3=vec![0.;q*d*d];let mut f4=f3.clone();for i in 0..d{for j in 0..d{for z in 0..q{f1[(i*d+j)*q+z]=u1[i*d+j+m*z]*s1[z].sqrt();f2[(i*d+j)*q+z]=u2[i*d+j+m*z]*s2[z].sqrt();f3[(z*d+i)*d+j]=s1[z].sqrt()*v1[z+m*(i*d+j)];f4[(z*d+i)*d+j]=s2[z].sqrt()*v2[z+m*(i*d+j)];}}}
 // P[(r,u),(w,b)] and Q[(w,b),(l,d)] in column-major order.
 let rows=q*q;let mid=d*d;let mut p=vec![0.;rows*mid];let mut v=vec![0.;mid*rows];for ww in 0..d{for bb in 0..d{let z=ww*d+bb;for r in 0..q{for u in 0..q{let mut sum=0.;for aa in 0..d{sum+=f1[(ww*d+aa)*q+r]*f2[(aa*d+bb)*q+u];}p[r*q+u+rows*z]=sum;}}for left in 0..q{for down in 0..q{let mut sum=0.;for gamma in 0..d{sum+=f3[(left*d+bb)*d+gamma]*f4[(down*d+gamma)*d+ww];}v[z+mid*(left*q+down)]=sum;}}}}
 let mut out=vec![0.;rows*rows];let (mi,ni,ki)=(rows as i32,rows as i32,mid as i32);let n=b'N' as c_char;unsafe{dgemm_(&n,&n,&mi,&ni,&ki,&1.,p.as_ptr(),&mi,v.as_ptr(),&ki,&0.,out.as_mut_ptr(),&mi);}let mut newa=vec![0.;q*q*q*q];for r in 0..q{for u in 0..q{for left in 0..q{for down in 0..q{newa[((r*q+u)*q+left)*q+down]=out[r*q+u+rows*(left*q+down)];}}}}
 steps.push(TrgStep{log_scale:scale.ln(),singular1:s1,singular2:s2,factors:vec![f1,f2,f3,f4],old_dim:d,new_dim:q});a=newa;d=q;count/=2.;
 }
 let mut z=0.;for i in 0..d{for j in 0..d{z+=a[((i*d+j)*d+i)*d+j];}}assert!(z>0.,"periodic closure {z}");((logscale+z.ln())/(l*l)as f64,a,steps)
}
