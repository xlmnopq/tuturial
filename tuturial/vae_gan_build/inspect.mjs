import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/jhg/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx'));
const dir='D:/tuturial/vae_gan_build';
await fs.writeFile(dir+'/template.json',JSON.stringify(p.toProto(),null,2));
await fs.writeFile(dir+'/template-inspect.ndjson',(await p.inspect({select:'$.slides[*]',include:['id','index','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i]; const b=await p.export({slide:s,format:'png',scale:0.6});
 await fs.writeFile(dir+`/source-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
console.log('rendered',p.slides.items.length);
