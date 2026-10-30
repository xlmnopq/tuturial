import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from 'file:///C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
try {
const dir='D:/tuturial/vae_gan_build';
const p=await PresentationFile.importPptx(await FileBlob.load('D:/tuturial/output/VAE与GAN_原理与对比.pptx'));
await fs.mkdir(dir+'/final-render',{recursive:true});
const changes=[];const hash=b=>createHash('sha256').update(b).digest('hex');
for(let i=0;i<p.slides.items.length;i++){
 const name=`slide-${String(i+1).padStart(2,'0')}.png`;
 const data=new Uint8Array(await (await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(dir+'/final-render/'+name,data);
 if(hash(data)!==hash(await fs.readFile(dir+'/'+name)))changes.push(i+1);
}
console.log(JSON.stringify({slideCount:p.slides.items.length,visuallyChanged:changes}));
}catch(e){console.error(e.message);process.exitCode=1;}
