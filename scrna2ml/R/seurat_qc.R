# Per-sample Seurat QC, provisional broad annotation and pseudobulk export.
# Inputs: a standardized library manifest, QC parameters as JSON; outputs one checkpointed folder per library
# (pseudobulk counts, cell-type composition, immune-compartment pseudobulk, QC summary).
suppressPackageStartupMessages({library(Seurat);library(Matrix);library(data.table);library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
manifest_path <- args[1]; out <- args[2]; qc_path <- args[3]
qc_all <- fromJSON(qc_path); qc <- qc_all[c('min_counts','min_features','max_percent_mt','min_cells_per_sample','variable_features','pca_dimensions','cluster_resolution','celltype_min_cells')]
set.seed(42);options(future.globals.maxSize=4*1024^3);future::plan('sequential')
manifest <- fread(manifest_path,na.strings=c('','NA'))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
markers <- list(T=c('CD3D','CD3E','TRAC'),NK=c('NKG7','GNLY','KLRD1'),B=c('MS4A1','CD79A','CD79B'),Plasma=c('MZB1','JCHAIN','SDC1'),Myeloid=c('LYZ','LST1','FCER1G'),DC=c('FCER1A','CD1C','CLEC10A'),Mast=c('TPSAB1','TPSB2','KIT'),Epithelial=c('EPCAM','KRT8','KRT18'),Endothelial=c('PECAM1','VWF','KDR'),Fibroblast=c('COL1A1','COL1A2','DCN'),Erythroid=c('HBB','HBA1','HBA2'),Platelet=c('PPBP','PF4','ITGA2B'))
immune_types <- c('T','NK','B','Plasma','Myeloid','DC','Mast')
signature_file<-tempfile();saveRDS(list(algorithm='seurat_v1_immune_export',qc=qc),signature_file,version=2)
signature<-unname(tools::md5sum(signature_file));unlink(signature_file)
checkpoint_valid<-function(folder){
 p<-file.path(folder,'complete.json');if(!file.exists(p))return(FALSE)
 identical(fromJSON(p)$signature,signature)
}
write_pb <- function(raw,idx,path) fwrite(data.table(gene=rownames(raw),count=Matrix::rowSums(raw[,idx,drop=FALSE])),path,sep='\t')
process <- function(counts,key) {
 set.seed(42)
 folder <- file.path(out,gsub('[^A-Za-z0-9_.-]','_',key));dir.create(folder,showWarnings=FALSE)
 if(checkpoint_valid(folder))return(invisible(NULL))
 unlink(list.files(folder,full.names=TRUE))
 input_cells <- ncol(counts)
 totals <- Matrix::colSums(counts); detected <- Matrix::colSums(counts>0)
 mt <- grepl('^MT-',rownames(counts));pct <- if(any(mt))100*Matrix::colSums(counts[mt,,drop=FALSE])/pmax(totals,1) else rep(NA_real_,ncol(counts))
 keep <- totals>=qc$min_counts & detected>=qc$min_features & (is.na(pct)|pct<=qc$max_percent_mt)
 counts <- counts[,keep,drop=FALSE]
 if(ncol(counts)<qc$min_cells_per_sample) {write_json(list(status='excluded',signature=signature,reason='Fewer than configured minimum QC-passing cells',input_cells=input_cells,retained=ncol(counts)),file.path(folder,'complete.json'),auto_unbox=TRUE);return(invisible(NULL))}
 obj<-CreateSeuratObject(counts=counts,min.cells=3,project=key)
 obj[['percent.mt']]<-if(any(grepl('^MT-',rownames(obj))))PercentageFeatureSet(obj,pattern='^MT-') else NA_real_
 obj<-NormalizeData(obj,verbose=FALSE)
 dat<-GetAssayData(obj,layer='data')
 scores<-sapply(markers,function(gs){gs<-intersect(gs,rownames(dat));if(length(gs)<2)rep(0,ncol(dat)) else Matrix::colMeans(dat[gs,,drop=FALSE])})
 best<-max.col(scores,ties.method='first');peak<-scores[cbind(seq_len(nrow(scores)),best)];second<-apply(scores,1,function(x)sort(x,decreasing=TRUE)[2])
 obj$celltype_provisional<-ifelse(peak>=0.5 & peak-second>=0.15,names(markers)[best],'Unknown')
 obj$annotation_margin<-peak-second
 # Upper count tails are flagged, never described as a validated doublet caller.
 obj$high_count_flag<-obj$nCount_RNA>exp(median(log(obj$nCount_RNA))+3*mad(log(obj$nCount_RNA)))
 obj<-FindVariableFeatures(obj,nfeatures=min(qc$variable_features,nrow(obj)-1),verbose=FALSE)
 obj<-ScaleData(obj,features=VariableFeatures(obj),verbose=FALSE)
 npcs<-min(qc$pca_dimensions,ncol(obj)-2,length(VariableFeatures(obj))-1)
 obj<-RunPCA(obj,npcs=npcs,verbose=FALSE)
 obj<-FindNeighbors(obj,dims=seq_len(npcs),verbose=FALSE)
 obj<-FindClusters(obj,resolution=qc$cluster_resolution,verbose=FALSE,random.seed=42)
 obj<-RunUMAP(obj,dims=seq_len(npcs),n.neighbors=min(30,ncol(obj)-1),verbose=FALSE,seed.use=42)
 pdf(file.path(folder,'qc_umap.pdf'),width=11,height=7)
 print(VlnPlot(obj,features=c('nFeature_RNA','nCount_RNA'),pt.size=0));print(DimPlot(obj,group.by='celltype_provisional',label=TRUE));dev.off()
 # Sample aggregation is the unit for ML; no cell-level labels enter supervised fitting.
 raw<-GetAssayData(obj,layer='counts'); md<-obj[[]]
 prefix<-file.path(folder,gsub('[^A-Za-z0-9_.-]','_',key))
 sel<-seq_len(ncol(raw))
 write_pb(raw,sel,paste0(prefix,'_pseudobulk.tsv'))
 ct<-table(factor(md$celltype_provisional,levels=c(names(markers),'Unknown')))
 fwrite(data.table(celltype=names(ct),count=as.integer(ct),proportion=as.numeric(ct)/sum(ct)),paste0(prefix,'_composition.tsv'),sep='\t')
 for(typ in names(ct)[ct>=qc$celltype_min_cells]) write_pb(raw,which(md$celltype_provisional==typ),paste0(prefix,'_celltype_',typ,'.tsv'))
 imm<-which(md$celltype_provisional %in% immune_types)
 if(length(imm)>0) write_pb(raw,imm,paste0(prefix,'_immune_pseudobulk.tsv'))
 ict<-ct[immune_types]
 fwrite(data.table(celltype=names(ict),count=as.integer(ict),proportion=if(sum(ict)>0) as.numeric(ict)/sum(ict) else 0),paste0(prefix,'_immune_composition.tsv'),sep='\t')
 fwrite(data.table(cell_id=rownames(md),md),file.path(folder,'cell_metadata.tsv.gz'),sep='\t')
 saveRDS(obj,file.path(folder,'seurat.rds'),compress=TRUE)
 write_json(list(status='complete',signature=signature,sample_id=key,input_cells=input_cells,retained=ncol(obj),immune_cells=length(imm),genes=nrow(obj),
   median_genes_per_cell=median(md$nFeature_RNA),median_umis_per_cell=median(md$nCount_RNA),median_percent_mt=if(any(mt))median(md$percent.mt) else NA,
   high_count_flagged=sum(md$high_count_flag),unknown_fraction=mean(md$celltype_provisional=='Unknown'),mitochondrial_genes_available=any(mt),
   doublet_method='upper-count flag only; no validated doublet removal',annotation='fixed broad markers; provisional',qc_parameters=qc,Seurat=as.character(packageVersion('Seurat'))),
   file.path(folder,'complete.json'),auto_unbox=TRUE,pretty=TRUE,na='null')
 rm(obj);gc()
}
read_features <- function(p) as.data.table(read.delim(if(grepl('\\.gz$',p)) gzfile(p) else p,header=FALSE,stringsAsFactors=FALSE,quote='',comment.char=''))
process_entry <- function(i){
 r<-manifest[i]
 if(checkpoint_valid(file.path(out,r$sample_id)))return(invisible(TRUE))
 message('Processing ',r$sample_id)
 counts<-as(readMM(r$matrix),'CsparseMatrix')
 genes<-read_features(r$genes)[[1]];bars<-read_features(r$barcodes)[[1]]
 if(length(genes)!=nrow(counts)||length(bars)!=ncol(counts))stop('Dimension mismatch for ',r$sample_id)
 rownames(counts)<-make.unique(as.character(genes));colnames(counts)<-as.character(bars)
 process(counts,r$sample_id)
 rm(counts);gc();invisible(TRUE)
}
workers<-if(is.null(qc_all$workers))1L else max(1L,as.integer(qc_all$workers))
results<-parallel::mclapply(seq_len(nrow(manifest)),function(i) tryCatch(process_entry(i),error=function(e){message('FAILED ',manifest$sample_id[i],': ',conditionMessage(e));FALSE}),mc.cores=workers,mc.preschedule=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,'R_sessionInfo.txt'))
if(!all(vapply(results,isTRUE,logical(1))))stop('One or more samples failed; rerun resumes from checkpoints')
