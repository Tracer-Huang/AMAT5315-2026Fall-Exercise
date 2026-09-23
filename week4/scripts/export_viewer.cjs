// Export the unmodified course viewer's own Save PNG output using a real browser.
// Usage: node scripts/export_viewer.cjs [week4-root] [evidence-directory]
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs/promises');
const path = require('node:path');
const root = path.resolve(process.argv[2] || '.');
const evidence = path.resolve(process.argv[3] || path.join(root, 'evidence'));
const mime = {'.html':'text/html; charset=utf-8','.jsonl':'application/x-ndjson','.png':'image/png','.json':'application/json'};

async function main() {
  await fs.mkdir(evidence,{recursive:true});
  const server=http.createServer(async(req,res)=>{
    const filename=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if (!filename.startsWith(root+path.sep)) {res.writeHead(403).end();return;}
    try {const data=await fs.readFile(filename);res.writeHead(200,{'Content-Type':mime[path.extname(filename)]||'text/plain'});res.end(data);}
    catch {res.writeHead(404).end('Not found');}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base=`http://127.0.0.1:${server.address().port}`;
  let browser;
  const receipt={viewer:'unmodified course viewer.html',browser:null,generated_by:'agent-driven browser; student independent review pending',frames:[]};
  try {
    browser=await chromium.launch({headless:true,...(process.env.BROWSER_CHANNEL?{channel:process.env.BROWSER_CHANNEL}:{})});
    receipt.browser=await browser.version();
    const page=await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    for (const [source,targets] of [
      ['artifacts/taylor-green/fields.jsonl',[[1,false,'viewer-taylor-green.png']]],
      ['fields.jsonl',[[0,false,'viewer-t0.png'],[2,false,'viewer-t2.png'],[5,false,'viewer-t5.png'],[10,false,'viewer-t10.png'],[10,true,'viewer-tracers-t10.png']]],
    ]) {
      await page.goto(`${base}/viewer.html?src=${encodeURIComponent(base+'/'+source)}`,{waitUntil:'networkidle'});
      await page.locator('#savepng').waitFor();
      for (const [t,tracers,name] of targets) {
        await page.locator('#scrub').fill(String(Math.round(t/.1)));
        if (await page.locator('#showtr').isEnabled()) await page.locator('#showtr').setChecked(tracers);
        const displayed=await page.locator('#ro-t').innerText();
        if (Math.abs(Number(displayed)-t)>1e-9) throw Error(`Wrong viewer time: ${displayed} != ${t}`);
        const download=page.waitForEvent('download');await page.locator('#savepng').click();
        await (await download).saveAs(path.join(evidence,name));
        receipt.frames.push({file:name,source,time:t,tracers,grid:await page.locator('#ro-n').innerText(),displayed_enstrophy:await page.locator('#ro-z').innerText()});
      }
      // Exercise actual playback and pause, not just static rendering.
      await page.locator('#scrub').fill('0');await page.locator('#play').click();
      await page.waitForFunction(()=>Number(document.querySelector('#scrub').value)>0);
      await page.locator('#play').click();
    }
    if(errors.length)throw Error(errors.join('\n'));
    receipt.playback='advanced and paused successfully for both recordings';
    receipt.page_errors=errors;
    await fs.writeFile(path.join(evidence,'viewer-export.json'),JSON.stringify(receipt,null,2)+'\n');
    console.log(`Exported ${receipt.frames.length} course viewer PNGs; playback passed.`);
  } finally {if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
}
main().catch(error=>{console.error(error);process.exitCode=1});
