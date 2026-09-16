"""Use the unmodified course viewer to export three stamped images, or verify public links.
Requires npx and Chrome. The normal reproduction uses a local recording; --verify-public
opens fresh browser contexts and checks each shared temperature without a login.
"""

import argparse
import functools
import http.server
import json
import pathlib
import subprocess
import threading
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
COURSE = "https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-public", metavar="RAW_RECORDING_URL")
    args = parser.parse_args()
    cli = [
        "npx",
        "--yes",
        "--package",
        "@playwright/cli@0.1.20",
        "playwright-cli",
        "-s=week3-course",
    ]
    logs = []
    server = None

    def call(*items):
        result = subprocess.run(
            cli + list(items), cwd=ROOT, text=True, capture_output=True, check=True
        )
        logs.append(result.stdout)
        if "### Error" in result.stdout:
            raise RuntimeError(result.stdout)

    try:
        if args.verify_public:
            call("open", COURSE)
            call("snapshot")
            code = """async (page) => {
                const raw=RAW;
                await page.goto(COURSE+'?src='+encodeURIComponent(raw));
                await page.waitForFunction(() => document.querySelector('#src').textContent.includes('410 frames'),null,{timeout:60000});
                await page.context().grantPermissions(['clipboard-read','clipboard-write'],{origin:new URL(COURSE).origin});
                const checked=[];
                for (const t of ['1.8','2.3','3.0']) {
                    await page.locator('#temperature').selectOption(t);
                    await page.locator('#copy-link').click();
                    await page.waitForFunction(want => {
                        const box=document.querySelector('#share-link');
                        return document.querySelector('#load-status').textContent.includes('Link copied for T = '+Number(want)+'.') ||
                          (!box.hidden && Number(new URL(box.value).searchParams.get('T'))===Number(want));
                    },t);
                    const url=await page.evaluate(async () => {
                        const box=document.querySelector('#share-link');
                        return box.hidden ? await navigator.clipboard.readText() : box.value;
                    });
                    if (new URL(url).searchParams.get('src')!==raw) throw new Error('wrong copied source');
                    const context=await page.context().browser().newContext();
                    try {
                        if ((await context.cookies()).length) throw new Error('context not fresh');
                        const p=await context.newPage();await p.goto(url);
                        await p.waitForFunction(() => document.querySelector('#src').textContent.includes('410 frames'),null,{timeout:60000});
                        const value=Number(await p.locator('#ro-T').textContent());
                        if (Math.abs(value-Number(t))>1e-9) throw new Error('wrong temperature '+value);
                        if (await p.locator('#copy-link').isDisabled()) throw new Error('public share link disabled');
                        checked.push({url,temperature:value,frames:await p.locator('#src').textContent(),freshContext:true});
                    } finally { await context.close(); }
                }
                return checked;
            }""".replace("COURSE", json.dumps(COURSE)).replace(
                "RAW", json.dumps(args.verify_public)
            )
            call("run-code", code)
        else:
            work = ROOT / "runs/viewer"
            work.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(COURSE) as response:
                (work / "week3-viewer.html").write_bytes(response.read())
            handler = functools.partial(
                http.server.SimpleHTTPRequestHandler, directory=str(ROOT)
            )
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            call(
                "open",
                f"http://127.0.0.1:{server.server_port}/runs/viewer/week3-viewer.html",
            )
            call("snapshot")
            code = """async (page) => {
                await page.locator('#file').setInputFiles(FILE);
                await page.waitForFunction(() => document.querySelector('#src').textContent.includes('410 frames'));
                for (const t of ['1.8','2.3','3.0']) {
                    await page.locator('#temperature').selectOption(t);
                    const event=page.waitForEvent('download');await page.locator('#savepng').click();
                    await (await event).saveAs(OUT+'/viewer-T'+t+'.png');
                }
                const before=await page.locator('#ro-frame').textContent();await page.locator('#play').click();
                await page.waitForFunction(old => document.querySelector('#ro-frame').textContent!==old,before);await page.locator('#play').click();
                return '410 frames loaded; three stamped images exported; playback advanced';
            }""".replace("FILE", json.dumps(str(ROOT / "spins.jsonl"))).replace(
                "OUT", json.dumps(str(ROOT / "evidence"))
            )
            call("run-code", code)
            call("snapshot")
        target = ROOT / "runs/analysis"
        target.mkdir(parents=True, exist_ok=True)
        (
            target / ("public-viewer.txt" if args.verify_public else "viewer.txt")
        ).write_text("\n".join(logs))
        print("\n".join(logs))
    finally:
        subprocess.run(cli + ["close"], cwd=ROOT, capture_output=True, check=False)
        if server is not None:
            server.shutdown()


if __name__ == "__main__":
    main()
