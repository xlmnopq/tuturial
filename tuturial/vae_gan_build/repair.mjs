import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
try {
const p=await PresentationFile.importPptx(await FileBlob.load('D:/tuturial/vae_gan_build/candidate.pptx'));
const hits=(await p.inspect({select:"$.slides[11].elements[?(@.kind == 'textbox')]",include:['id','text']})).records;
const target=hits.find(x=>x.text.includes('同时查看损失'));
if(!target)throw new Error('Expected text missing');
p.resolve(target.id).text.replace('同时查看损失、样本质量和多样性。','检查样本质量与多样性。');
const slide=p.slides.items[11];
await fs.writeFile('D:/tuturial/vae_gan_build/slide-12.png',new Uint8Array(await (await p.export({slide,format:'png',scale:1})).arrayBuffer()));
await (await PresentationFile.exportPptx(p)).save('D:/tuturial/vae_gan_build/candidate-v2.pptx');
console.log('Repair exported');
}catch(e){console.error(e.message);process.exitCode=1;}
