import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {finalizePresentation} from 'file:///C:/Users/jhg/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const DIR='D:/tuturial/vae_gan_build';
const SKILL='C:/Users/jhg/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const REF='C:/Users/jhg/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
try {
const r=await finalizePresentation({workspaceDir:'D:/tuturial',candidatePath:DIR+'/candidate-v2.pptx',finalPath:'D:/tuturial/output/VAE与GAN_原理与对比.pptx',pythonExecutable:'C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','14'],requiredNativeTableOwnerSlides:[2,14],fontPolicy:{basis:'reference',families:['Helvetica Neue','Helvetica Neue Medium'],referencePath:REF,referenceSha256:createHash('sha256').update(await fs.readFile(REF)).digest('hex'),scriptFonts:{ea:'Microsoft YaHei'}},verifyArtifactToolImport:true,receiptPath:DIR+'/validation.json'});
console.log(JSON.stringify(r));
}catch(e){console.error(e.message);if(e.details)console.error(JSON.stringify(e.details));process.exitCode=1;}

