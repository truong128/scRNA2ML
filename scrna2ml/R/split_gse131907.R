# Split the GSE131907 combined raw UMI matrix (dense integer data.frame, ~208k cells) into per-sample gzipped MTX.
suppressPackageStartupMessages({library(Matrix);library(data.table)})
args <- commandArgs(trailingOnly=TRUE)
rds <- args[1]; cells_path <- args[2]; out <- args[3]
# The GEO file is an (already compressed) .rds wrapped in an extra gzip layer.
tmp_rds <- tempfile(fileext='.rds')
if(system2('sh', c('-c', shQuote(paste('gzip -dc', shQuote(rds), '>', shQuote(tmp_rds))))) != 0) stop('gunzip failed')
m <- readRDS(tmp_rds); unlink(tmp_rds)
cells <- fread(cells_path)
missing <- setdiff(cells$Index, colnames(m))
if(length(missing) > 0) stop(length(missing), ' annotated cells absent from the UMI matrix')
genes <- rownames(m)
for(s in unique(cells$Sample)){
  d <- file.path(out, s); dir.create(d, recursive=TRUE, showWarnings=FALSE)
  ids <- cells$Index[cells$Sample == s]
  dense <- as.matrix(m[, ids, drop=FALSE])
  if(!is.numeric(dense) || any(dense < 0) || any(dense != round(dense))) stop('GSE131907 sample ', s, ' is not raw integer counts')
  sub <- as(dense, 'CsparseMatrix'); rm(dense)
  tmp <- file.path(d, 'matrix.mtx'); writeMM(sub, tmp)
  if(system2('gzip', c('-f', shQuote(tmp))) != 0) stop('gzip failed for ', s)
  writeLines(genes, gzfile(file.path(d, 'genes.tsv.gz')))
  writeLines(ids, gzfile(file.path(d, 'barcodes.tsv.gz')))
  writeLines(sprintf('{"genes": %d, "cells": %d, "nnz": %d}', nrow(sub), ncol(sub), length(sub@x)), file.path(d, 'complete.json'))
  message(s, ': ', ncol(sub), ' cells')
}
