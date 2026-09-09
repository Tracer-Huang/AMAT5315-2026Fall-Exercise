"""Encode chronological real browser captures as a silent time-compressed demo."""
import argparse,bisect,json,subprocess
from pathlib import Path
from PIL import Image
import imageio_ffmpeg

parser=argparse.ArgumentParser()
parser.add_argument("manifest",type=Path)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--receipt",type=Path,required=True)
parser.add_argument("--max-seconds",type=float,default=110)
args=parser.parse_args()
rows=[json.loads(line) for line in args.manifest.read_text().splitlines()]
assert len(rows)>1 and all(rows[i+1]["requested_ms"]>rows[i]["requested_ms"] for i in range(len(rows)-1))
start=rows[0]["requested_ms"]
end=rows[-1]["completed_ms"]
source_duration=(end-start)/1000
speed=max(1,source_duration/args.max_seconds)
sizes=[Image.open(row["file"]).size for row in rows]
assert len(set(sizes))==1,"viewport dimensions changed during capture"
source_width,source_height=sizes[0]
width=(source_width+1)//2*2
height=(source_height+1)//2*2
args.out.parent.mkdir(parents=True,exist_ok=True)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
fps=15
output_frames=round(source_duration/speed*fps)
command=[ffmpeg,"-y","-loglevel","warning","-f","rawvideo","-pixel_format","rgb24",
         "-video_size",f"{source_width}x{source_height}","-framerate",str(fps),"-i","pipe:0",
         "-vf",f"pad={width}:{height}:0:0:black,setsar=1",
         "-an","-c:v","libx264","-preset","medium","-crf","18","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(args.out)]
# Explicit constant-frame-rate resampling avoids sub-frame ffconcat timestamp
# rounding. Each output frame is an unaltered captured image, held until the
# next capture in the uniformly compressed source timeline.
process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
times=[row["requested_ms"] for row in rows]
previous_index=None
pixels=None
used=set()
for frame_index in range(output_frames):
    source_time=start+frame_index/fps*speed*1000
    index=max(0,bisect.bisect_right(times,source_time)-1)
    if index!=previous_index:
        pixels=Image.open(rows[index]["file"]).convert("RGB").tobytes()
        previous_index=index
        used.add(index)
    process.stdin.write(pixels)
process.stdin.close()
error=process.stderr.read().decode()
if process.wait()!=0:raise RuntimeError(error)
reader=imageio_ffmpeg.read_frames(str(args.out));meta=next(reader);reader.close()
assert 0<meta["duration"]<120
args.receipt.parent.mkdir(parents=True,exist_ok=True)
args.receipt.write_text(json.dumps({'source_manifest':str(args.manifest),'source_capture_count':len(rows),
    'source_elapsed_seconds':source_duration,'uniform_time_compression':speed,
    'encoded_duration_seconds':meta['duration'],'encoded_frames':output_frames,'fps':fps,
    'encoded_unique_captures':len(used),'dimensions':[width,height],'audio':False,
    'method':'Actual browser screenshots resampled chronologically at constant fps with uniform time compression; real fresh-shell command output shown in a read-only browser panel. No fabricated UI or physics values. One-pixel padding only; no screenshot content resizing.',
    'file':str(args.out),'bytes':args.out.stat().st_size,'encoder_command':command},indent=2)+'\n')
print(args.receipt.read_text())
