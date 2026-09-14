import { cp, mkdir, rm } from 'node:fs/promises';
import { resolve } from 'node:path';
const root=resolve(new URL('..',import.meta.url).pathname);const dist=resolve(root,'dist');await rm(dist,{recursive:true,force:true});await mkdir(dist,{recursive:true});for(const item of ['index.html','styles.css','src'])await cp(resolve(root,item),resolve(dist,item),{recursive:true});console.log('Built static dist/ successfully.');
