import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from 'file:///C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/jhg/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations/container_tools/artifact_tool_utils.mjs';
const DIR='D:/tuturial/vae_gan_build';
const SKILL='C:/Users/jhg/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const REF='C:/Users/jhg/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.1/skills/artifact-template-simple-light-mode/assets/reference.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(REF));
const originals=[...p.slides.items];
const originalIds=(await p.inspect({select:'$.slides[*]',include:['id']})).records.map(x=>x.id);
const result=[];
const SRC={vae:'Kingma & Welling, Auto-Encoding Variational Bayes, ICLR 2014. https://arxiv.org/abs/1312.6114',vae2:'Kingma & Welling, An Introduction to Variational Autoencoders, 2019. https://arxiv.org/abs/1906.02691',gan:'Goodfellow et al., Generative Adversarial Nets, NeurIPS 2014. https://arxiv.org/abs/1406.2661',tutorial:'Goodfellow, NIPS 2016 Tutorial: Generative Adversarial Networks. https://arxiv.org/abs/1701.00160',dcgan:'Radford et al., DCGAN, ICLR 2016. https://arxiv.org/abs/1511.06434',fid:'Heusel et al., TTUR and FID, NeurIPS 2017. https://arxiv.org/abs/1706.08500'};
function runs(text,size,bold=false){return [...text.matchAll(/[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]+|[^\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]+/gu)].map(m=>({run:m[0],style:{typeface:/[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef]/u.test(m[0])?'Microsoft YaHei':'Helvetica Neue',fontSize:`${size}px`,bold,color:'#000000'}}));}
function set(shape,text,size=24,bold=false){shape.text.style={typeface:'Helvetica Neue',fontSize:size,color:'#000000',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};shape.text=text.split('\n').map(t=>({runs:runs(t,size,bold),bulletCharacter:'',marginLeft:0,indent:0,spaceAfter:600}));}
function block(shape,heading,paras,{head=28,body=24}={}){shape.text.style={fontSize:body,typeface:'Helvetica Neue',color:'#000000',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};shape.text=[{runs:runs(heading,head),bulletCharacter:'',marginLeft:0,indent:0,spaceAfter:1600},...paras.map(t=>({runs:runs(t,body),bulletCharacter:'',marginLeft:0,indent:0,spaceAfter:1400}))];}
async function clone(n,title,notes,sources=[]){const s=originals[n-1].duplicate();result.push(s);
 const all=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records;
 const r=all.find(x=>p.resolve(x.id).id===s.id);if(!r)throw new Error('Cannot locate duplicated slide');
 const els=r.elements;
 const txt=els.filter(x=>x.kind==='textbox');
 if(n!==1&&n!==26){const t=txt.find(x=>x.position.top<80);set(p.resolve(t.id),title,38.67);const num=txt.find(x=>/^\d+$/.test(x.text??'')&&x.position.top>650);if(num)set(p.resolve(num.id),String(result.length),13.33);}
 s.speakerNotes.text=notes+'\n\n参考来源：\n'+sources.map(x=>SRC[x]).join('\n');
 return {s,els,txt,r};}
async function two(title,h1,b1,h2,b2,notes,sources=[]){const a=await clone(5,title,notes,sources);const b=a.txt.filter(x=>x.position.top>150&&x.position.top<650).sort((x,y)=>x.position.left-y.position.left);block(p.resolve(b[0].id),h1,b1);block(p.resolve(b[1].id),h2,b2);return a;}
async function four(title,items,notes,sources=[]){const a=await clone(13,title,notes,sources);const b=a.txt.filter(x=>x.position.top>150&&x.position.top<650).sort((x,y)=>Math.round(x.position.top/100)-Math.round(y.position.top/100)||x.position.left-y.position.left);items.forEach((x,i)=>block(p.resolve(b[i].id),x[0],x[1],{head:28,body:24}));return a;}
async function imageSlide(title,heading,paras,file,notes,sources){const a=await clone(8,title,notes,sources);const body=a.txt.find(x=>x.position.top>150&&x.position.top<650);block(p.resolve(body.id),heading,paras,{head:28,body:24});const e=a.els.find(x=>x.kind==='image');const im=p.resolve(e.id);im.replace({blob:new Uint8Array(await fs.readFile(DIR+'/'+file)),contentType:'image/png',alt:heading+'，AI 生成的教学示意图，非实验结果',fit:'contain'});im.frame=e.position;im.crop={left:0,top:0,right:0,bottom:0};im.fit='contain';im.geometry='rect';im.borderRadius=0;return a;}

// 01 Cover, using the retained cover without changing geometry.
{
const a=await clone(1,'','建议讲解时长约 20–25 分钟。面向了解神经网络与梯度下降的初学者。开场用手写数字举例：模型学习大量数字图像后，能否生成新的合理数字？本讲解以经典 VAE 与原始 GAN 为范围，说明常见实现与局限，避免把它们的性质推广到所有变体。',['vae','gan']);
const byY=[...a.txt].sort((x,y)=>x.position.top-y.position.top);
set(p.resolve(byY[0].id),'生成模型入门',24);
set(p.resolve(byY[1].id),'VAE 与 GAN',96);
set(p.resolve(byY[2].id),'变分自编码器与生成对抗网络\n原理、训练目标与方法对比',24);
}
// 02 Agenda.
{
const a=await clone(3,'讲解路线','每一部分围绕一个可以复述的概念展开。先建立共同语言，再分别介绍 VAE 与 GAN，最后比较它们并讨论怎样验证模型。');const t=p.resolve(a.els.find(x=>x.kind==='table').id);const vals=['生成模型与潜变量','VAE：概率编码与重构','VAE：重参数化与损失','GAN：生成器与判别器','GAN：交替训练与常见问题','方法对比、应用与练习'];
for(let i=0;i<6;i++){t.getCell(i,1).text.style={typeface:'Helvetica Neue',fontSize:32,color:'#000000'};t.cells.set(i,1,[{runs:runs(vals[i],32),bulletCharacter:''}]);}
}
await two('生成模型的任务','判别式建模',['输入一张图像 x，预测类别 y。','典型目标：学习 p(y | x)。','例子：判断这张手写数字是不是 3。'],'生成式建模',['学习数据的分布或采样机制。','采样后得到新的图像 x。','例子：生成一张训练集中没有的 3。'],'区分“判断类别”和“生成样本”。生成模型可以使用带标签或不带标签的数据。这里讨论的是用神经网络学习数据的生成机制。GAN 通过隐式采样建模，通常不直接给出可计算的样本似然。',['vae2','tutorial']);
await two('潜变量 z 与数据 x','潜变量的直觉',['z 是控制生成结果的隐藏表示。','它可能编码笔画倾斜、粗细等变化。','这些语义通常需要实验验证。'],'共同的生成过程',['先从简单分布采样 z。','再用解码器或生成器得到 x。','改变 z，可以改变生成的样本。'],'z 的维数、先验以及网络结构都由建模者选择。标准正态分布是常用选择，不是唯一选择。不要承诺某个维度必然只代表一个人类语义特征。VAE 的生成模型还指定 pθ(x|z)，GAN 通常直接用确定性 G(z) 产生样本。',['vae2','gan']);
await imageSlide('VAE 的模型结构','编码一个分布',['编码器 qφ(z | x) 输出均值 μ 和标准差 σ。','从这个分布采样 z，再由解码器 pθ(x | z) 重构输入。','生成新样本时，从先验 p(z) 采样，直接使用解码器。'],'vae.png','VAE 全称 Variational Autoencoder。常见设定 qφ(z|x)=N(μ,diag(σ²))，先验 p(z)=N(0,I)。编码器负责近似后验推断，解码器负责条件生成。右图仅解释计算关系，手写数字并非训练输出。图由内置 imagegen 生成，使用黑白 VAE 编码器、分布参数、采样与解码器结构提示词。',['vae']);
await two('重参数化技巧','可微的采样表达',['z = μ + σ ⊙ ε','ε ~ N(0, I)','⊙ 表示逐元素相乘。'],'梯度为什么能传回编码器',['噪声 ε 独立于编码器参数。','对固定 ε，z 是 μ 和 σ 的可微函数。','重构损失可以更新编码器和解码器。'],'直接把随机采样当作普通确定性计算，无法直接套用常规的路径导数。重参数化把随机性移到 ε 中，给出随机梯度估计。实现时通常输出 log σ²，再计算 σ=exp(0.5 log σ²)，以保持正值与数值稳定。它不会消除随机性。',['vae']);
await two('VAE 的训练目标：ELBO','最大化证据下界',['ELBO = E_q[log pθ(x | z)]','               − KL(qφ(z | x) ∥ p(z))','log pθ(x) ≥ ELBO'],'最小化损失的写法',['L = 重构负对数似然 + KL 散度','重构项：保留输入信息。','KL 项：约束近似后验接近先验。'],'期望下标 q 指 qφ(z|x)。ELBO 是 evidence lower bound，即边缘对数似然的下界。标准 VAE 中 KL 的权重是 1。重构项具体形式由观测分布决定：Bernoulli 常对应二元交叉熵，固定方差 Gaussian 与平方误差相关。不能把任何 MSE 加 KL 都无条件称为正确似然目标。对角高斯与标准正态的 KL 为 0.5 Σ(μ²+σ²−log σ²−1)。',['vae']);
await two('VAE 的训练与生成','训练阶段',['1  输入训练样本 x。','2  计算 μ、σ，采样 z 并重构 x。','3  合并重构项与 KL 项，反向传播。'],'生成阶段',['1  采样 z ~ N(0, I)。','2  用解码器产生分布参数或样本。','3  重复采样，得到不同结果。'],'训练使用编码器近似推断输入对应的潜变量。无条件生成使用先验，不需要先输入一张原图。应区分重构和随机生成：重构好，并不保证从先验采样也好。训练时应明确重构项在像素维度求和还是求均值，否则会改变它与 KL 项的相对权重。',['vae2']);
await four('VAE 的特点与局限',[
['便于表示学习',['编码器为每个输入提供潜变量分布。']],
['具有概率目标',['ELBO 支持近似推断与生成模型学习。']],
['可能丢失细节',['简单像素似然与模型容量会影响清晰度。']],
['可能出现后验坍塌',['解码器忽略 z，近似后验接近先验。']]
],'“VAE 一定模糊”是过度概括。输出质量受似然、模型结构与优化影响。后验坍塌常表现为 KL 很小、潜变量对生成作用弱，但 KL 小本身不足以完成诊断。可以检查潜变量利用率，并对 KL 权重调度或解码器结构开展控制实验。',['vae2']);
await imageSlide('GAN 的模型结构','两个网络共同训练',['生成器 G 把随机噪声 z 映射成样本。','判别器 D 区分真实样本和生成样本。','G 利用 D 的梯度反馈改进生成结果。'],'gan.png','GAN 全称 Generative Adversarial Network。D(x) 在原始 GAN 中表示输入来自真实数据的概率。训练中 D 和 G 同时改善，但按步骤交替更新。生成阶段仅保留 G。右图是内置 imagegen 生成的黑白双数据流教学示意，不代表实际训练结果。',['gan']);
await two('GAN 的训练目标','原始极小极大目标',['min_G max_D V(D, G)','V = E_x[log D(x)]','      + E_z[log(1 − D(G(z)))]'],'常用的生成器损失',['L_G = −E_z[log D(G(z))]','D 希望真实样本得分高，生成样本得分低。','G 希望生成样本获得更高的真实得分。'],'E_x 对真实数据分布取期望，E_z 对噪声先验取期望。右侧是非饱和生成器损失，常用于避免训练早期梯度过弱，与原始 minimax 的生成器目标并不相同。在理想的无限容量与全局最优条件下，生成分布等于真实分布，最优 D 为 1/2。实际训练 D 接近 1/2 不能单独证明成功。',['gan']);
{
const a=await clone(6,'GAN 的交替训练','生成器和判别器分别有自己的参数和优化器。更新 D 时将 G 的生成样本从 G 的计算图中分离。更新 G 时固定 D 的参数，但仍保留 D 对输入的梯度。更新比例与学习率属于需要调节的超参数。',['gan','tutorial']);
const b=a.txt.filter(x=>x.position.top>150&&x.position.top<650).sort((x,y)=>x.position.left-y.position.left);
block(p.resolve(b[0].id),'01  更新判别器',['真实样本记为 1，生成样本记为 0。','固定 G，最小化二分类损失。']);
block(p.resolve(b[1].id),'02  更新生成器',['重新采样噪声并生成图像。','固定 D 参数，用其梯度更新 G。']);
block(p.resolve(b[2].id),'03  重复并检查',['交替更新两组参数。','同时查看损失、样本质量和多样性。']);
}
await four('GAN 的常见训练问题',[
['模式坍塌',['不同噪声产生很相似的样本，覆盖的类型减少。']],
['训练失衡',['D 过强或过弱，都可能削弱对 G 的有效反馈。']],
['振荡与不稳定',['两个网络的目标互相变化，损失可能反复波动。']],
['损失不等于质量',['单个损失下降，不能证明生成效果改善。']]
],'模式坍塌可以通过固定噪声样本、批次多样性与类别覆盖等手段辅助发现。损失只反映当前对手下的训练目标。后续方法通过结构、正则化或目标改进稳定性，但没有一种技巧能保证所有任务都成功。DCGAN 给出了卷积结构的经验约束。',['tutorial','dcgan']);
// Editable comparison table, retaining the reference's table position and visual style.
{
const a=await clone(14,'VAE 与 GAN 的核心区别','这张表比较经典 VAE 与原始 GAN。带编码器的 GAN、VAE-GAN 混合模型，以及更复杂的 VAE 都可能改变这些性质。样本质量和覆盖率必须通过一致的实验设定比较。',['vae2','tutorial']);
const intro=a.txt.find(x=>x.position.top>90&&x.position.top<180);set(p.resolve(intro.id),'比较范围：经典 VAE 与原始 GAN，具体表现取决于结构和训练设置。',24);
const t=p.resolve(a.els.find(x=>x.kind==='table').id);t.deleteColumns({index:3,count:2});t.deleteRows({index:7,count:2});
const vals=[['比较维度','VAE','GAN'],['学习目标','最大化 ELBO','优化对抗目标'],['模型组件','编码器 + 解码器','生成器 + 判别器'],['输入反推 z','自带近似后验编码器','原始模型没有编码器'],['生成过程','先验采样后解码','噪声输入 G'],['主要挑战','细节损失、后验坍塌','模式坍塌、训练不稳定'],['适合关注','潜表示与概率推断','样本视觉逼真度']];
t.columnWidths=[230,483.5,483.5];
for(let r=0;r<vals.length;r++){for(let c=0;c<3;c++){t.getCell(r,c).text.style={typeface:'Helvetica Neue',fontSize:r===0?23:22,color:'#000000',bold:r===0};t.cells.set(r,c,[{runs:runs(vals[r][c],r===0?23:22,r===0),bulletCharacter:''}]);}t.rows[r].height=58;}
}
await two('同一个手写数字任务','VAE 的观察重点',['查看输入与重构图的差异。','比较后验采样和先验采样的结果。','检查插值是否连续、z 是否有效。'],'GAN 的观察重点',['固定一组噪声，记录训练中的变化。','检查生成数字的清晰度与类型覆盖。','关注重复样本和训练波动。'],'这是一份实验设计，而非实验结果。可在 MNIST 上使用同样的训练数据划分、图像归一化和相近计算预算。不要直接比较 VAE 损失与 GAN 损失的数值。固定噪声便于跟踪模型变化，独立随机采样便于观察总体多样性。',['vae2','tutorial']);
await four('方法选择与应用',[
['潜变量分析与重构',['VAE 提供输入的编码分布，便于分析潜表示。']],
['图像合成',['GAN 常用于追求视觉逼真度的生成任务。']],
['条件生成',['向网络加入类别 y，可控制目标类别。']],
['具体项目评估',['结合任务指标、样本覆盖和训练成本选择。']]
],'这些是依据模型结构给出的选择思路，不是绝对排名。条件 VAE 将 q(z|x,y) 与 p(x|z,y) 纳入模型，条件 GAN 将 y 提供给生成器和判别器。重构与异常检测等任务的效果需要专门验证，不能仅依据模型名称判断。',['vae2','dcgan','tutorial']);
await four('生成模型的评估',[
['质量',['图像结构是否合理，细节是否符合任务要求？']],
['多样性',['生成结果是否覆盖不同类别和变化？']],
['定量指标',['FID 比较真实与生成图像的特征分布。']],
['公平设置',['统一数据划分、预处理、样本量和评估协议。']]
],'FID 使用预训练特征表示中的高斯近似来比较分布，数值较低通常更好，但其结果受特征提取器、样本量和预处理影响。标准 ImageNet Inception 特征对手写数字等不同领域不一定合适，应说明评估器的适用性。FID 无法单独完成记忆、偏差与多样性的全部检查。',['fid']);
await two('需要记住的关键点','VAE',['用概率编码器推断 z。','通过重参数化传播梯度。','用重构项与 KL 项联合训练。'],'GAN',['让 G 和 D 形成对抗训练。','交替更新，两者都影响学习过程。','同时评估样本质量与多样性。'],'让听众分别用一句话复述：VAE 用变分推断学习潜变量生成模型，GAN 用判别器提供的信号训练生成器。接着追问：VAE 如何生成新图，GAN 为什么不能只看 D 的准确率？答案分别是从先验采样再解码，以及准确率会受到当前对手与训练失衡影响。',['vae','gan']);
await two('参考文献与延伸阅读','VAE',['Kingma & Welling, 2014','Auto-Encoding Variational Bayes','arxiv.org/abs/1312.6114','Kingma & Welling, 2019','An Introduction to Variational Autoencoders','arxiv.org/abs/1906.02691'],'GAN 与评估',['Goodfellow et al., 2014','Generative Adversarial Nets','arxiv.org/abs/1406.2661','Goodfellow, 2016 教程：1701.00160','Radford et al., DCGAN：1511.06434','Heusel et al., FID：1706.08500'],'建议先读原始论文中的模型结构、训练算法与实验设置，再阅读教程。完整可点击链接见各页备注。VAE 预印本于 2013 年发布，本文按 ICLR 2014 会议年份引用。GAN 教程的演讲发生于 NIPS 2016，预印本版本后续于 2017 年修订。',['vae','vae2','gan','tutorial','dcgan','fid']);
{
const a=await clone(26,'','讨论题答案：1. 约束近似后验接近先验，减小训练时的潜表示与生成时先验采样之间的不匹配。2. 模式坍塌或覆盖不足。3. 只需从先验采样 z，并使用 VAE 解码器或 GAN 生成器。可让听众先回答，再回到模型结构图检查。',['vae2','tutorial']);
const byY=[...a.txt].sort((x,y)=>x.position.top-y.position.top);
set(p.resolve(byY[0].id),'回顾与讨论',24);set(p.resolve(byY[1].id),'三个理解检查',85.33);
const b=p.resolve(byY[2].id);b.position={...byY[2].position,width:1080,height:140};set(b,'VAE 为什么要约束潜变量分布？\nGAN 生成的图很清晰，但都很相似，说明什么？\n两种模型在生成阶段分别需要哪些网络？',24);
}

p.delete(originalIds);
const canonicalSlides=(await p.inspect({select:'$.slides[*]',include:['id']})).records;
p.slides.reorder(result.map(s=>canonicalSlides.find(x=>p.resolve(x.id).id===s.id).id));
await fs.writeFile(DIR+'/authored.json',JSON.stringify(p.toProto(),null,2));
await fs.writeFile(DIR+'/content.ndjson',(await p.inspect({select:'$.slides[*]',include:['id','title','elements.id','elements.text','elements.position','elements.style','notes.text']})).ndjson);
await (await PresentationFile.exportPptx(p)).save(DIR+'/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];const b=await p.export({slide:s,format:'png',scale:1});
 await fs.writeFile(DIR+`/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await b.arrayBuffer()));
 await fs.writeFile(DIR+`/slide-${String(i+1).padStart(2,'0')}.layout.json`,await (await s.export({format:'layout'})).text());
}
if(process.argv.includes('--skip-finalize'))process.exit(0);
const hash=createHash('sha256').update(await fs.readFile(REF)).digest('hex');
const final=await finalizePresentation({workspaceDir:'D:/tuturial',candidatePath:DIR+'/candidate.pptx',finalPath:'D:/tuturial/output/VAE与GAN_原理与对比.pptx',pythonExecutable:'C:/Users/jhg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],requiredNativeTableOwnerSlides:[2,14],fontPolicy:{basis:'reference',families:['Helvetica Neue','Helvetica Neue Medium'],referencePath:REF,referenceSha256:hash,scriptFonts:{ea:'Microsoft YaHei'}},verifyArtifactToolImport:true,receiptPath:DIR+'/validation.json'});
console.log(JSON.stringify(final));
