"""MP4 rendering helper embedded in the Rust executable; physics runs in Rust."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amat5315-md-matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def radial_distribution(positions, box, edges):
    """Unordered-pair counts normalized by N*rho*annulus area/2."""
    d=positions[:,None,:]-positions[None,:,:]
    d-=box*np.round(d/box)
    distance=np.sqrt(np.einsum("ijk,ijk->ij",d,d))
    pairs=distance[np.triu_indices(len(positions),1)]
    counts=np.histogram(pairs,bins=edges)[0]
    area=np.pi*np.diff(edges**2)
    rho=len(positions)/np.prod(box)
    return 2*counts/(len(positions)*rho*area)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run=json.loads((args.directory/"run.json").read_text())
    frames=[json.loads(line) for line in (args.directory/"traj.jsonl").read_text().splitlines()]
    if not frames: raise ValueError("no production frames")
    box=np.array(run["box"],dtype=float)
    edges=np.linspace(0,box.min()/2,105)
    radii=(edges[:-1]+edges[1:])/2
    rdf=np.array([radial_distribution(np.array(frame["pos"]),box,edges) for frame in frames])
    cumulative=np.vstack([np.zeros(len(radii)),np.cumsum(rdf,axis=0)])
    ffmpeg=os.environ.get("MD_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["animation.ffmpeg_path"]=ffmpeg
    fig,(atoms,structure)=plt.subplots(1,2,figsize=(9.6,4.5),layout="constrained")
    p=np.array(frames[0]["pos"])
    scatter=atoms.scatter(p[:,0],p[:,1],s=45*(100/run["n"]),color="#3d8dab",edgecolors="none")
    atoms.set(xlim=(0,box[0]),ylim=(0,box[1]),aspect="equal",xlabel="x",ylabel="y")
    line,=structure.plot(radii,rdf[0],color="#215f9a",linewidth=1.5)
    structure.axhline(1,color="#777777",linestyle="--",linewidth=0.8)
    structure.set(xlim=(0,edges[-1]),ylim=(0,max(5,float(np.percentile(rdf,99.8))*1.1)),
                  xlabel="Pair distance r",ylabel="g(r)",title="Pair structure | recent 20 frames")
    structure.grid(alpha=0.15)
    title=atoms.set_title("")
    fps=20
    duration=len(frames)/fps
    bitrate=max(30,min(850,int(1_500_000*8/duration/1000)))
    writer=FFMpegWriter(fps=fps,codec="libx264",bitrate=bitrate,
                        extra_args=["-pix_fmt","yuv420p","-movflags","+faststart",
                                    "-maxrate",f"{bitrate}k","-bufsize",f"{2*bitrate}k"])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with writer.saving(fig,str(args.out),dpi=100):
        for index,frame in enumerate(frames):
            start=max(0,index-19)
            g=(cumulative[index+1]-cumulative[start])/(index+1-start)
            scatter.set_offsets(np.array(frame["pos"]))
            line.set_ydata(g)
            velocity=np.array(frame["vel"])
            t_speed=float(np.mean(np.sum(velocity**2,axis=1))/2)
            title.set_text(f'{run["n"]} atoms | t={frame["t"]:.2f} | T={t_speed:.3f}')
            writer.grab_frame()
    plt.close(fig)
    size=args.out.stat().st_size
    if size>=2_000_000: raise RuntimeError(f"video {size} bytes exceeds the 2 MB limit")
    print(f"wrote {args.out} ({len(frames)} frames, {duration:g} s, {size} bytes)")


if __name__=="__main__":
    main()
